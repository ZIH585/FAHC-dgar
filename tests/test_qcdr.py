"""QCDR CPU contracts: scores, branch gradients, real RGCN wiring and resume.

All facts/tensors are synthetic. No dataset epochs or CUDA initialization.
The independent reference loads BOTH decoder and RGCN from the clean baseline.
"""
import contextlib
import copy
import io
from pathlib import Path
import pickle
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from src.cli import build_arg_parser, validate_args
from src.decoder import ConvTransE
from src.qcdr import decouple_candidate_gradient, effective_candidate_alpha
from src.rrgcn import RecurrentRGCN
from src.runtime import save_training_state, restore_training_state, settings
from rgcn.utils import build_sub_graph

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'a4f00eb'
torch.set_num_threads(1)


def reference_module(path):
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'],
                                     cwd=ROOT, text=True)
    module = types.ModuleType('baseline_' + Path(path).stem)
    module.__file__ = str(ROOT / path)
    exec(compile(source, f'{BASELINE}:{path}', 'exec'), module.__dict__)
    return module


def args_for(mode='original', alpha=1.0, replay_weight=1.0):
    return validate_args(build_arg_parser().parse_args([
        '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8',
        '--n-bases', '2', '--n-basis', '2', '--add-static-graph', 'false',
        '--diffusion_steps', '2', '--qcdr-mode', mode,
        '--qcdr-candidate-alpha', str(alpha),
        '--qcdr-replay-entity-weight', str(replay_weight)]))


def make_model(args, cls=RecurrentRGCN, static=False):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls('convtranse', 'uvrgcn', 8, 2, 1 if static else 0,
                   2 if static else 0, 8, 2, 'sub', 3,
                   num_bases=2, num_basis=2, num_hidden_layers=2,
                   dropout=args.dropout, self_loop=True, layer_norm=True,
                   input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
                   feat_dropout=args.feat_dropout, weight=args.weight,
                   discount=args.discount, angle=args.angle, use_static=static,
                   entity_prediction=True, relation_prediction=True,
                   use_cuda=False, gpu=-1, args=args)


def graph():
    return build_sub_graph(8, 2, np.array([[0, 0, 1], [2, 1, 3], [1, 1, 4],
                                         [4, 0, 5], [1, 1, 6]], dtype=np.int64), False, -1)


def fixture(replay=True, empty=False, singleton=False, static=False):
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]])[:1 if singleton else 2]
    tag = torch.tensor([[4, 0, 5], [1, 1, 6]])[:1 if singleton else 2]
    rep = torch.sin(torch.arange(len(tag) * 24).float()).reshape(len(tag), 3, 8)
    if empty:
        tag, rep = tag[:0], rep[:0]
    sg = (build_sub_graph(10, 1, np.array([[0, 0, 8], [1, 0, 9], [2, 0, 8]]), False, -1)
          if static else None)
    return ([graph(), graph()], triples, sg, False), dict(
        model_output=rep if replay else None,
        model_output_x_t=rep[:, -1] if replay else None,
        tag=tag, diffuc='add', output_reply_triple=tag)


def total_loss(model, **kw):
    positional, keywords = fixture(**kw)
    losses = model.get_loss(*positional, **keywords)
    return .7 * losses[0] + .3 * losses[1] + losses[2], losses


def gradients(model):
    return {name: p.grad.detach().clone() if p.grad is not None else None
            for name, p in model.named_parameters()}


class RoutingTests(unittest.TestCase):
    def test_independent_query_and_candidate_gradients(self):
        torch.default_generator.manual_seed(21)
        q0, e0 = torch.randn(3, 4), torch.randn(7, 4)
        outputs = {}
        for alpha in (1., 0., .25, .5):
            q, e = q0.clone().requires_grad_(), e0.clone().requires_grad_()
            scores = q @ decouple_candidate_gradient(e, alpha).t()
            F.cross_entropy(scores, torch.tensor([1, 2, 3])).backward()
            outputs[alpha] = (scores.detach(), q.grad, e.grad)
        original = outputs[1.]
        self.assertGreater(float(original[2].norm()), 0.)
        for alpha, (scores, qgrad, egrad) in outputs.items():
            self.assertTrue(torch.equal(scores, original[0]))
            torch.testing.assert_close(qgrad, original[1], rtol=0, atol=0)
            self.assertGreater(float(qgrad.norm()), 0.)
            if alpha == 0:
                self.assertIsNone(egrad)
            else:
                torch.testing.assert_close(egrad, alpha * original[2], rtol=1e-6, atol=1e-7)

    def test_endpoints_dtype_device_empty_and_no_grad(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            x = torch.tensor([[-.8, .3, 1.2]], dtype=dtype, requires_grad=True)
            self.assertIs(decouple_candidate_gradient(x, 1.), x)
            for alpha in (0., .25, .5, 1.):
                y = decouple_candidate_gradient(x, alpha)
                self.assertTrue(torch.equal(x, y))
                self.assertEqual((y.dtype, y.device, y.shape), (x.dtype, x.device, x.shape))
                with torch.no_grad():
                    score = decouple_candidate_gradient(x, alpha) @ x.t()
                self.assertFalse(score.requires_grad)
        self.assertEqual(decouple_candidate_gradient(torch.empty(0, 8), .25).shape, (0, 8))
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                decouple_candidate_gradient(torch.ones(1), alpha)

    def test_decoder_forward_and_independent_branch_gradients(self):
        torch.default_generator.manual_seed(23)
        decoder = ConvTransE(8, 8, .2, .2, .2)
        initial = copy.deepcopy(decoder.state_dict())
        query0, relation0, bank0 = torch.randn(8, 8), torch.randn(4, 8), torch.randn(8, 8)
        for batch_size in (1, 3):
            triples = torch.tensor([[0, 0, 1], [2, 1, 3], [0, 2, 4]])[:batch_size]
            out = {}
            for alpha in (1., 0., .25, .5):
                decoder.load_state_dict(initial)
                decoder.zero_grad(set_to_none=True)
                torch.default_generator.manual_seed(31)
                query, relation, bank = [v.clone().requires_grad_() for v in (query0, relation0, bank0)]
                scores = decoder(query, relation, triples, partial_embeding=bank,
                                 candidate_grad_scale=alpha)
                F.cross_entropy(scores, triples[:, 2]).backward()
                out[alpha] = (scores.detach(), query.grad, relation.grad, bank.grad, gradients(decoder))
            for alpha, result in out.items():
                for i in (0, 1, 2):
                    torch.testing.assert_close(result[i], out[1.][i], rtol=0, atol=0)
                self.assertGreater(float(result[1].norm()), 0.)
                if alpha == 0:
                    self.assertIsNone(result[3])
                else:
                    torch.testing.assert_close(result[3], alpha * out[1.][3], rtol=1e-6, atol=1e-7)
                for name, grad in result[4].items():
                    if grad is not None:
                        torch.testing.assert_close(grad, out[1.][4][name], rtol=0, atol=0)
                self.assertGreater(float(result[4]['conv1.weight'].norm()), 0.)

    def test_shared_entity_gradient_decomposes_into_query_plus_scaled_candidate(self):
        torch.default_generator.manual_seed(34)
        decoder = ConvTransE(8, 8).eval()
        e0, relation = torch.randn(8, 8), torch.randn(4, 8)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]])
        grads = {}
        for alpha in (0., .25, .5, 1.):
            entities = e0.clone().requires_grad_()
            loss = F.cross_entropy(decoder(entities, relation, triples,
                                           candidate_grad_scale=alpha), triples[:, 2])
            grads[alpha], = torch.autograd.grad(loss, entities)
        entities = e0.clone().requires_grad_()
        candidate_only = F.cross_entropy(
            decoder(entities.detach(), relation, triples, partial_embeding=torch.tanh(entities)),
            triples[:, 2])
        candidate_grad, = torch.autograd.grad(candidate_only, entities)
        self.assertGreater(float(grads[0.][0].norm()), 0.)
        self.assertEqual(float(grads[0.][7].norm()), 0.)
        self.assertGreater(float(candidate_grad[7].norm()), 0.)
        for alpha in (0., .25, .5, 1.):
            torch.testing.assert_close(grads[alpha], grads[0.] + alpha * candidate_grad,
                                       rtol=2e-5, atol=2e-7)

    def test_cpu_autocast(self):
        torch.default_generator.manual_seed(37)
        decoder = ConvTransE(8, 8).eval()
        q0, c0, r = torch.randn(8, 8), torch.randn(8, 8), torch.randn(4, 8)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]])
        outputs = {}
        for alpha in (1., 0., .25, .5):
            q, c = q0.clone().requires_grad_(), c0.clone().requires_grad_()
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                scores = decoder(q, r, triples, partial_embeding=c, candidate_grad_scale=alpha)
                loss = F.cross_entropy(scores, triples[:, 2])
            loss.backward()
            outputs[alpha] = (scores.detach(), q.grad, c.grad)
        for alpha, (scores, qgrad, cgrad) in outputs.items():
            self.assertEqual(scores.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(scores, outputs[1.][0]))
            torch.testing.assert_close(qgrad, outputs[1.][1], rtol=0, atol=0)
            if alpha == 0:
                self.assertIsNone(cgrad)
            else:
                torch.testing.assert_close(cgrad, alpha * outputs[1.][2], rtol=1e-6, atol=1e-7)


class ModelTests(unittest.TestCase):
    def test_replay_weight_scales_both_entity_terms_only(self):
        results = {}
        for weight in (0., .5, 1.):
            torch.default_generator.manual_seed(71)
            args = args_for(replay_weight=weight)
            args.mu = .3
            model = make_model(args, static=True).eval()
            calls = []
            original = model.decoder_ob.forward
            def score(entities, relations, triples, *a, **kw):
                scores = original(entities, relations, triples, *a, **kw)
                calls.append((scores, triples))
                return scores
            with mock.patch.object(model.decoder_ob, 'forward', side_effect=score):
                total, losses = total_loss(model, static=True)
            self.assertEqual([len(t) for _, t in calls], [4, 2, 4])
            current = F.cross_entropy(calls[0][0], calls[0][1][:, 2])
            replay = (F.cross_entropy(calls[1][0], calls[1][1][:, 2])
                      + args.mu * F.cross_entropy(calls[2][0], calls[2][1][:, 2]))
            torch.testing.assert_close(losses[0], (current + weight * replay).reshape(1))
            other_grads = torch.autograd.grad(losses[1] + losses[2], tuple(model.parameters()),
                                             retain_graph=True, allow_unused=True)
            total.backward()
            results[weight] = (gradients(model), other_grads, [v.detach() for v in losses[:4]])
        for weight, (grads, other_grads, losses) in results.items():
            for name, g1 in results[1.][0].items():
                g0, gw = results[0.][0][name], grads[name]
                if g1 is None:
                    self.assertIsNone(gw)
                else:
                    torch.testing.assert_close(gw, g0 + weight * (g1 - g0), rtol=3e-5, atol=3e-7)
            for gw, g1 in zip(other_grads, results[1.][1]):
                if g1 is not None:
                    torch.testing.assert_close(gw, g1, rtol=0, atol=0)
            for i in (1, 2, 3):
                torch.testing.assert_close(losses[i], results[1.][2][i], rtol=0, atol=0)

    def test_replay_weight_scales_query_and_candidate_gradients(self):
        outputs = {}
        torch.default_generator.manual_seed(73)
        q0, r0, bank0 = torch.randn(8, 8), torch.randn(4, 8), torch.randn(8, 8)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]])
        for weight in (.5, 1.):
            torch.default_generator.manual_seed(75)
            model = make_model(args_for(replay_weight=weight)).eval()
            query, relations, bank = [x.clone().requires_grad_() for x in (q0, r0, bank0)]
            original = model.decoder_ob.forward
            scores = []
            def score(e, r, triples, **kw):
                result = original(e, r, triples, partial_embeding=bank, **kw)
                scores.append(result.detach())
                return result
            with mock.patch.object(model.decoder_ob, 'forward', side_effect=score):
                loss = model.loss_of_reply(query, relations, triples, mode='e')
            loss.backward()
            outputs[weight] = (scores[0], loss.detach(), query.grad, relations.grad, bank.grad)
        self.assertTrue(torch.equal(outputs[.5][0], outputs[1.][0]))
        for i in (1, 2, 3, 4):
            self.assertGreater(float(outputs[1.][i].norm()), 0.)
            torch.testing.assert_close(outputs[.5][i], .5 * outputs[1.][i], rtol=1e-6, atol=1e-7)

    def test_original_and_scaled_one_match_independent_baseline_through_adam(self):
        old_decoder, old_rgcn = reference_module('src/decoder.py'), reference_module('src/rrgcn.py')
        old_rgcn.ConvTransE, old_rgcn.ConvTransR = old_decoder.ConvTransE, old_decoder.ConvTransR
        models = []
        for cls, mode in ((old_rgcn.RecurrentRGCN, 'original'),
                          (RecurrentRGCN, 'original'), (RecurrentRGCN, 'candidate-scaled')):
            torch.default_generator.manual_seed(41)
            models.append(make_model(args_for(mode), cls, static=True))
        initial = copy.deepcopy(models[0].state_dict())
        outputs, grads = [], []
        for model in models:
            self.assertEqual(set(model.state_dict()), set(initial))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            torch.default_generator.manual_seed(43)
            total, losses = total_loss(model, static=True)
            total.backward()
            grads.append(gradients(model))
            outputs.append([v.detach().clone() for v in losses])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        for i in (1, 2):
            for a, b in zip(outputs[0], outputs[i]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name, value in grads[0].items():
                if value is None:
                    self.assertIsNone(grads[i][name])
                else:
                    torch.testing.assert_close(value, grads[i][name], rtol=0, atol=0)
            for name, value in models[0].state_dict().items():
                torch.testing.assert_close(value, models[i].state_dict()[name], rtol=0, atol=0)

    def test_real_get_loss_routes_both_replay_terms_only(self):
        model = make_model(args_for('candidate-stopgrad'))
        entity_calls, relation_calls = [], []
        original_e, original_r = model.decoder_ob.forward, model.rdecoder.forward
        def entity(e, r, triples, *a, **kw):
            entity_calls.append((triples.clone(), kw.get('candidate_grad_scale', 1.)))
            return original_e(e, r, triples, *a, **kw)
        def relation(e, r, triples, *a, **kw):
            relation_calls.append(kw.copy())
            return original_r(e, r, triples, *a, **kw)
        with mock.patch.object(model.decoder_ob, 'forward', side_effect=entity), \
             mock.patch.object(model.rdecoder, 'forward', side_effect=relation):
            total, _ = total_loss(model)
            total.backward()
        self.assertEqual([scale for _, scale in entity_calls], [1., 0., 0.])
        self.assertEqual([len(t) for t, _ in entity_calls], [4, 2, 4])
        self.assertEqual(entity_calls[2][0][:, 1].tolist(), [0, 1, 2, 3])
        self.assertEqual(len(relation_calls), 2)
        self.assertTrue(all('candidate_grad_scale' not in kw for kw in relation_calls))

    def test_full_model_gradient_affinity_current_and_relation_unchanged(self):
        results = {}
        for alpha in (0., .25, .5, 1.):
            torch.default_generator.manual_seed(47)
            model = make_model(args_for('candidate-scaled', alpha))
            torch.default_generator.manual_seed(49)
            total, losses = total_loss(model)
            relation_grads = torch.autograd.grad(losses[1], tuple(model.parameters()),
                                                retain_graph=True, allow_unused=True)
            total.backward()
            results[alpha] = (losses[0].detach(), gradients(model), relation_grads)
        for alpha, (loss, grads, relation_grads) in results.items():
            self.assertTrue(torch.equal(loss, results[1.][0]))
            for name, g1 in results[1.][1].items():
                g0, ga = results[0.][1][name], grads[name]
                if g1 is None:
                    self.assertIsNone(ga)
                else:
                    torch.testing.assert_close(ga, g0 + alpha * (g1 - g0), rtol=3e-5, atol=3e-7)
            for ga, g1 in zip(relation_grads, results[1.][2]):
                if g1 is not None:
                    torch.testing.assert_close(ga, g1, rtol=0, atol=0)
        self.assertGreater(float((results[1.][1]['dynamic_emb'] - results[0.][1]['dynamic_emb']).norm()), 0.)

    def test_replay_query_still_reaches_shared_encoder_and_decoder(self):
        torch.default_generator.manual_seed(53)
        model = make_model(args_for('candidate-stopgrad')).eval()
        positional, kw = fixture()
        h, _, r, _, _ = model.forward(positional[0], None, False,
                                      model_output=kw['model_output'], model_output_x_t=kw['model_output_x_t'],
                                      tag=kw['tag'], diffu_rep='add')
        model.loss_of_reply(F.normalize(h[-1]), r, kw['tag'], mode='e').backward()
        actual = gradients(model)
        for prefix in ('dynamic_emb', 'emb_rel', 'rgcn.', 'relation_cell_1.',
                       'time_gate_weight', 'decoder_ob.conv1.', 'decoder_ob.fc.'):
            norm = sum(float(g.norm()) for name, g in actual.items() if name.startswith(prefix) and g is not None)
            self.assertGreater(norm, 0., prefix)

    def test_no_replay_empty_singleton_and_first_stage(self):
        for kind in ({'replay': False}, {'empty': True}, {'singleton': True}):
            results = []
            for mode, alpha in (('original', 1.), ('candidate-stopgrad', 1.), ('candidate-scaled', 0.)):
                torch.default_generator.manual_seed(59)
                model = make_model(args_for(mode, alpha))
                torch.default_generator.manual_seed(61)
                loss, _ = total_loss(model, **kind)
                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                results.append((loss.detach(), gradients(model)))
            for result in results[1:]:
                self.assertTrue(torch.equal(result[0], results[0][0]))
            # No historical CE: all current gradients must match Original.
            left = results[0 if not kind.get('singleton') else 1][1]
            for name, g in results[2][1].items():
                if g is not None:
                    torch.testing.assert_close(g, left[name], rtol=0, atol=0)

    def test_prediction_bypasses_routing_and_is_identical(self):
        outputs = []
        for mode, alpha in (('original', 1.), ('candidate-stopgrad', 1.), ('candidate-scaled', .25)):
            torch.default_generator.manual_seed(67)
            model = make_model(args_for(mode, alpha)).eval()
            positional, kw = fixture(singleton=True)
            with mock.patch('src.decoder.decouple_candidate_gradient', side_effect=AssertionError('evaluation routing')):
                outputs.append(model.predict(positional[0], 2, None, positional[1],
                                             kw['model_output'], kw['model_output_x_t'], kw['tag'], False))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        for a in outputs[1:]:
            for x, y in zip(a, outputs[0]):
                self.assertTrue(torch.equal(x, y))
                self.assertFalse(x.requires_grad)
        self.assertEqual(outputs[0][1].shape, (2, 8))
        self.assertEqual(outputs[0][2].shape, (2, 4))


class RuntimeTests(unittest.TestCase):
    def test_cli_defaults_ranges_and_legacy_defaults(self):
        current = build_arg_parser().parse_args([])
        old = reference_module('src/cli.py').build_arg_parser().parse_args([])
        self.assertEqual({k: v for k, v in vars(current).items() if not k.startswith('qcdr_')}, vars(old))
        self.assertEqual((current.qcdr_mode, current.qcdr_candidate_alpha, current.qcdr_target),
                         ('original', 1., 'entity'))
        self.assertEqual(current.qcdr_replay_entity_weight, 1.)
        self.assertEqual(effective_candidate_alpha('original', .25), 1.)
        self.assertEqual(effective_candidate_alpha('candidate-stopgrad', 1.), 0.)
        for alpha in (0., .25, .5, 1.):
            self.assertEqual(args_for('candidate-scaled', alpha).qcdr_candidate_alpha, alpha)
        for alpha in (-1., 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                args_for('candidate-scaled', alpha)
            with self.assertRaises(ValueError):
                args_for(replay_weight=alpha)
        with contextlib.redirect_stderr(io.StringIO()):
            for tokens in (['--qcdr-mode', 'unknown'], ['--qcdr-target', 'both'], ['--qcdr-target', 'relation']):
                with self.assertRaises(SystemExit):
                    build_arg_parser().parse_args(tokens)

    def test_resume_without_extra_qcdr_state_and_reject_config_change(self):
        args = args_for('candidate-scaled', .25)
        model, teacher = make_model(args), make_model(args)
        diffu, diff_teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        opt, dopt = torch.optim.Adam(model.parameters()), torch.optim.Adam(diffu.parameters())
        loss, _ = total_loss(model)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        saved_model = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            args.run_dir = Path(temp)
            args._provenance = {'settings': settings(args)}
            save_training_state(args, 2, model, diffu, opt, dopt, 'cpu', auxiliary=None)
            args.resume = args.run_dir / 'latest.pt'
            expected_rng = torch.rand(5)
            with torch.no_grad():
                model.dynamic_emb.add_(1.)
            stage, auxiliary = restore_training_state(args, model, diffu, teacher, diff_teacher,
                                                       opt, dopt, 'cpu')
            self.assertEqual((stage, auxiliary), (3, None))
            self.assertTrue(torch.equal(torch.rand(5), expected_rng))
            self.assertEqual(model.qcdr_candidate_alpha, .25)
            self.assertFalse(any('qcdr' in name for name in model.state_dict()))
            for name, value in saved_model.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
                self.assertTrue(torch.equal(value, teacher.state_dict()[name]))
            self.assertGreater(len(opt.state), 0)
            args._provenance['settings']['qcdr_candidate_alpha'] = .5
            with self.assertRaises(ValueError):
                restore_training_state(args, model, diffu, teacher, diff_teacher, opt, dopt, 'cpu')

    def test_real_training_caller_one_synthetic_replay_batch_then_abort(self):
        import src.main as main
        args = args_for('candidate-stopgrad')
        data = types.SimpleNamespace(num_nodes=8, num_rels=2,
            train=np.array([[0, 0, 1, 0]] * 10), valid=np.array([[0, 1, 2, 1]] * 10),
            test=np.array([[0, 0, 3, 2]] * 10))
        seen = []
        original_loss = RecurrentRGCN.get_loss
        class BatchComplete(Exception):
            pass
        def generate(_self, _args, stage, *_positional, **_keywords):
            self.assertEqual(stage, 2)
            _, kw = fixture(singleton=True)
            # The injected replay triple really belongs to synthetic stage 0.
            return torch.tensor([[0, 0, 3]]), kw['model_output'], torch.tensor([[0, 0, 1]])
        def stop_after_backward(model, *positional, **keywords):
            self.assertEqual(model.qcdr_candidate_alpha, 0.)
            with mock.patch.object(model.decoder_ob, 'forward', wraps=model.decoder_ob.forward) as call:
                losses = original_loss(model, *positional, **keywords)
            self.assertEqual([c.kwargs.get('candidate_grad_scale', 1.) for c in call.call_args_list], [1., 0., 0.])
            (.7 * losses[0] + .3 * losses[1] + losses[2]).backward()
            self.assertGreater(float(model.dynamic_emb.grad.norm()), 0.)
            seen.append(True)
            raise BatchComplete()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as stdout, \
             contextlib.redirect_stderr(io.StringIO()):
            args.data_root, args.run_dir = Path(temp) / 'data', Path(temp) / 'run'
            args.resume = args.run_dir / 'latest.pt'
            directory = args.data_root / args.dataset
            cache = directory / 'history_snap_v2'; cache.mkdir(parents=True)
            for name in ('train', 'valid', 'test', 'stat'):
                (directory / f'{name}.txt').write_text('synthetic fixture\n')
            for stage, row in enumerate(((0, 0, 1), (0, 1, 2), (0, 0, 3))):
                (cache / f'snap_{stage}.pkl').write_bytes(pickle.dumps({'0': {row}}))
            with mock.patch.object(main.utils, 'load_data', return_value=data), \
                 mock.patch.object(main, 'fitlog'), \
                 mock.patch.object(main, 'restore_training_state', return_value=(2, None)), \
                 mock.patch.object(main.Att_Diffuse_model, 'gereration_feature', generate), \
                 mock.patch.object(RecurrentRGCN, 'get_loss', stop_after_backward), \
                 mock.patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer step allowed')), \
                 self.assertRaises(BatchComplete):
                main.run_experiment(args)
            self.assertEqual(stdout.getvalue().count('QCDR mode: candidate-stopgrad'), 1)
            self.assertIn('QCDR candidate alpha: 0.0', stdout.getvalue())
            self.assertFalse((args.run_dir / 'model.pt').exists())
            self.assertFalse((args.run_dir / 'latest.pt').exists())
        self.assertEqual(seen, [True])

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
