"""CPU mechanism, real DGAR wiring, baseline equivalence and resume checks."""
import ast
import contextlib
import copy
import io
import json
import pickle
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.grb import BalanceConfig, ReplayBalancer, weight_from_moments
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng, save_training_state, restore_training_state
from scripts.summarize_grb import summarize

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '23dd43dd914a783731c7f39e3fd66491d2aba429'


def make_model(cls=RecurrentRGCN):
    args = validate_args(build_arg_parser().parse_args([
        '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8',
        '--add-static-graph', 'true', '--n-bases', '2', '--n-layers', '1',
        '--mu', '.3', '--mu_r', '.6']))
    with contextlib.redirect_stdout(io.StringIO()):
        model = cls('convtranse', 'uvrgcn', 6, 2, 1, 2, 8, 2,
            'sub', sequence_len=3, num_bases=2, num_hidden_layers=1,
            dropout=.2, self_loop=True, layer_norm=True,
            input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
            entity_prediction=True, relation_prediction=True,
            use_static=True, weight=.5, discount=1, angle=10, args=args)
    return model


def inputs(batch=2, replay=True):
    current = torch.tensor([[0, 0, 1], [2, 1, 3]])[:batch]
    triples = np.array([[0, 0, 2], [1, 1, 3], [4, 0, 5]])
    history = [build_sub_graph(6, 2, triples, False, -1)]
    static = build_sub_graph(8, 1, np.array([[0, 0, 6], [1, 0, 7]]), False, -1)
    tag = torch.tensor([[0, 0, 2], [1, 1, 3]])[:batch]
    generated = torch.randn(len(tag), 3, 8)
    keywords = dict(model_output=generated if replay else None,
                    model_output_x_t=generated[:, -1] if replay else None,
                    tag=tag if replay else None, diffuc='add',
                    output_reply_triple=tag if replay else None)
    return (history, current, static, False), keywords


def run_loss(model, positional, keywords, state, controller=None):
    restore_rng(state, 'cpu')
    model.train()
    model.zero_grad()
    losses = model.get_loss(*positional, **keywords, replay_balancer=controller)
    total = .7*losses[0] + .3*losses[1] + losses[2]
    total.backward()
    gradients = {name: value.grad.clone() if value.grad is not None else None
                 for name, value in model.named_parameters()}
    return losses, gradients, rng_state('cpu')


class GrbTests(unittest.TestCase):
    def test_cli_defaults_and_invalid_parameters(self):
        parser = build_arg_parser()
        args = validate_args(parser.parse_args([]))
        self.assertEqual(BalanceConfig.from_args(args), BalanceConfig())
        for flags in (['--grb-min-weight', 'nan'], ['--grb-fixed-weight', '1.1'],
                      ['--grb-target-ratio', '0'], ['--grb-conflict-scale', '-1'],
                      ['--grb-ema', '1'], ['--grb-mode', 'adaptive', '--diffuc', 'false'],
                      ['--grb-mode', 'adaptive', '--task-weight', '0']):
            with self.assertRaises(ValueError):
                validate_args(parser.parse_args(flags))

    def test_gradient_geometry_scale_invariance_and_safe_fallback(self):
        config = BalanceConfig(mode='adaptive')
        cases = [([1., 1., 1.], 1.), ([1., 1., -1.], .5),
                 ([1., 4., 0.], .5), ([4., 1., 2.], 1.)]
        for values, expected in cases:
            for scale in (1., 1e-12, 1e8):
                weight, cosine, valid = weight_from_moments(torch.tensor(values)*scale, config)
                self.assertTrue(valid)
                self.assertAlmostEqual(float(weight), expected)
                self.assertTrue(torch.isfinite(cosine))
        for values in ([0., 1., 0.], [1., 0., 0.], [float('nan'), 1., 0.],
                       [float('inf'), 1., 0.]):
            weight, cosine, valid = weight_from_moments(torch.tensor(values), config)
            self.assertEqual(float(weight), 1.)
            self.assertFalse(valid)
            self.assertEqual(float(cosine), 0.)

    def test_weight_is_detached_and_both_replay_terms_are_scaled(self):
        h = torch.tensor([1., 2.], requires_grad=True)
        cur, forward, inverse = h[0]**2, 3*h[1]**2, .3*h.sum()**2
        original = ((torch.zeros(1) + cur) + forward) + inverse
        controller = ReplayBalancer(BalanceConfig(mode='adaptive', ema=0))
        controller.start_stage(1)
        result = controller.combine(original, cur, forward, inverse, h)
        self.assertIsNone(h.grad)
        record = controller._records[-1]
        weight = record[0]
        self.assertEqual(float(weight), .5)
        self.assertFalse(weight.requires_grad)
        self.assertFalse(controller._moments.requires_grad)
        self.assertTrue(all(not row.requires_grad and row.grad_fn is None for row in controller._records))
        result.backward()
        expected = torch.tensor([2., 0.]) + weight*(torch.tensor([0., 12.]) + torch.tensor([1.8, 1.8]))
        torch.testing.assert_close(h.grad, expected)

    def test_ema_reset_stage_records_and_no_replay(self):
        controller = ReplayBalancer(BalanceConfig(mode='adaptive'))
        with self.assertRaises(RuntimeError):
            controller.combine(torch.tensor(1.), None, None, None, torch.zeros(1))
        controller.start_stage(1)
        original = torch.tensor([2.], requires_grad=True)
        self.assertIs(controller.combine(original, None, None, None, original), original)
        h = torch.tensor([1., 2.], requires_grad=True)
        cur, a, b = h[0]**2, h[1]**2, h[1]**2
        controller.combine(cur+a+b, cur, a, b, h)
        self.assertIsNotNone(controller._moments)
        with tempfile.TemporaryDirectory() as tmp:
            report = controller.write_stage(Path(tmp))
            self.assertEqual(report['skipped_updates'], 1)
            self.assertEqual(report['measured_updates'], 1)
            self.assertEqual(json.loads((Path(tmp)/'grb-stages.jsonl').read_text())['stage'], 1)
            self.assertEqual(controller._records, [])
        controller.start_stage(2)
        self.assertIsNone(controller._moments)
        self.assertEqual(controller._skipped, 0)

    def test_zero_signal_fallback_does_not_bias_first_valid_ema(self):
        controller = ReplayBalancer(BalanceConfig(mode='adaptive'))
        controller.start_stage(1)
        h = torch.tensor([1., 2.], requires_grad=True)
        zero, a, b = 0*h.sum(), h[1]**2, h[1]**2
        original = zero+a+b
        first = controller.combine(original, zero, a, b, h)
        torch.testing.assert_close(first, original, rtol=0, atol=0)
        self.assertEqual(float(controller._records[-1][0]), 1.)
        self.assertFalse(controller._initialized)
        cur = h[0]**2
        controller.combine(cur+a+b, cur, a, b, h)
        torch.testing.assert_close(controller._moments, torch.tensor([4., 64., 0.]), rtol=0, atol=0)
        self.assertTrue(controller._initialized)
        previous = controller._moments.clone()
        controller.combine(original, zero, a, b, h)
        torch.testing.assert_close(controller._moments, previous, rtol=0, atol=0)
        self.assertEqual(float(controller._records[-1][0]), 1.)

    def test_real_off_and_fixed_one_match_independent_original_exactly(self):
        original_source = subprocess.run(['git', 'show', f'{BASELINE}:src/rrgcn.py'],
            cwd=ROOT, check=True, text=True, capture_output=True).stdout
        namespace = {'__name__': 'grb_reference'}
        exec(compile(original_source, str(ROOT/'src/rrgcn.py'), 'exec'), namespace)
        old_cls = namespace['RecurrentRGCN']
        old = make_model(old_cls)
        off, fixed = make_model(), make_model()
        off.load_state_dict(old.state_dict())
        fixed.load_state_dict(old.state_dict())
        positional, keywords = inputs()
        state = rng_state('cpu')
        expected = None
        for model, mode in ((old, None), (off, None), (fixed, 'fixed')):
            restore_rng(state, 'cpu')
            controller = ReplayBalancer(BalanceConfig(mode='fixed', fixed_weight=1)) if mode else None
            if controller:
                controller.start_stage(2)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            losses = model.get_loss(*positional, **keywords,
                **({'replay_balancer': controller} if model is not old else {}))
            total = .7*losses[0] + .3*losses[1] + losses[2]
            total.backward()
            grads = {name: value.grad.clone() if value.grad is not None else None
                     for name, value in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            result = ([x.detach().clone() for x in losses], grads,
                      copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), rng_state('cpu'))
            if expected is None:
                expected = result
            else:
                for a, b in zip(result[0], expected[0]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for name, grad in result[1].items():
                    if grad is None:
                        self.assertIsNone(expected[1][name])
                    else:
                        torch.testing.assert_close(grad, expected[1][name], rtol=0, atol=0)
                for name, value in result[2].items():
                    torch.testing.assert_close(value, expected[2][name], rtol=0, atol=0)
                for index, opt_state in result[3]['state'].items():
                    for key, value in opt_state.items():
                        torch.testing.assert_close(value, expected[3]['state'][index][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(result[4]['torch'], expected[4]['torch']))
                self.assertEqual(result[4]['python'], expected[4]['python'])

    def test_real_adaptive_signal_matches_manual_gradient_and_keeps_buffers_rng(self):
        model = make_model()
        manual, adaptive = copy.deepcopy(model), copy.deepcopy(model)
        positional, keywords = inputs(batch=1)
        state = rng_state('cpu')
        captures = {}
        ce = manual.loss_e.forward
        def capture_ce(scores, targets):
            loss = ce(scores, targets)
            captures.setdefault('entity_losses', []).append(loss)
            return loss
        decoder = manual.decoder_ob.forward
        def capture_repr(embedding, *args, **kw):
            captures['h'] = embedding
            return decoder(embedding, *args, **kw)
        restore_rng(state, 'cpu')
        with mock.patch.object(manual.loss_e, 'forward', capture_ce), \
             mock.patch.object(manual.decoder_ob, 'forward', capture_repr):
            manual_losses = manual.get_loss(*positional, **keywords)
        current, forward, reverse = captures['entity_losses']
        inverse = manual.args.mu*reverse
        gc, = torch.autograd.grad(current, captures['h'], retain_graph=True)
        gr, = torch.autograd.grad(forward+inverse, captures['h'], retain_graph=True)
        moments = torch.stack((gc.square().sum(), gr.square().sum(), (gc*gr).sum()))
        config = BalanceConfig(mode='adaptive', ema=0)
        weight, _, _ = weight_from_moments(moments, config)
        expected_e = (torch.zeros_like(manual_losses[0])+current) + weight*forward + weight*inverse
        expected_total = .7*expected_e + .3*manual_losses[1] + manual_losses[2]
        expected_total.backward()
        manual_rng = rng_state('cpu')
        controller = ReplayBalancer(config)
        controller.start_stage(2)
        result, gradients, after = run_loss(adaptive, positional, keywords, state, controller)
        torch.testing.assert_close(controller._moments, moments.detach(), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(result[0], expected_e, rtol=0, atol=0)
        for i in (1, 2, 3, 4):
            torch.testing.assert_close(result[i], manual_losses[i], rtol=0, atol=0)
        for name, parameter in manual.named_parameters():
            if parameter.grad is None:
                self.assertIsNone(gradients[name])
            else:
                torch.testing.assert_close(gradients[name], parameter.grad, rtol=1e-6, atol=1e-7)
        for name, value in manual.state_dict().items():
            torch.testing.assert_close(adaptive.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(torch.equal(after['torch'], manual_rng['torch']))
        self.assertEqual(after['python'], manual_rng['python'])
        self.assertEqual(len(controller._records), 1)

    def test_fixed_weight_changes_only_entity_replay_and_handles_empty(self):
        original = make_model()
        positional, keywords = inputs(batch=1)
        state = rng_state('cpu')
        results = {}
        for weight in (0., .5, 1.):
            model = copy.deepcopy(original)
            controller = ReplayBalancer(BalanceConfig(mode='fixed', fixed_weight=weight))
            controller.start_stage(2)
            results[weight] = run_loss(model, positional, keywords, state, controller)
        for name, grad in results[.5][1].items():
            if grad is not None:
                torch.testing.assert_close(grad, .5*(results[0.][1][name] + results[1.][1][name]),
                                           rtol=3e-5, atol=3e-7)
        for index in (1, 2, 3, 4):
            torch.testing.assert_close(results[.5][0][index], results[1.][0][index], rtol=0, atol=0)
        # First/no-history-replay stage and empty replay use exactly current loss.
        for empty in (False, True):
            positional, keywords = inputs(batch=1, replay=False)
            if empty:
                keywords.update(model_output=torch.empty(0, 3, 8), tag=torch.empty(0, 3, dtype=torch.long),
                                model_output_x_t=torch.empty(0, 8), output_reply_triple=torch.empty(0, 3, dtype=torch.long))
            left, right = copy.deepcopy(original), copy.deepcopy(original)
            controller = ReplayBalancer(BalanceConfig(mode='adaptive'))
            controller.start_stage(1)
            a = run_loss(left, positional, keywords, state)
            b = run_loss(right, positional, keywords, state, controller)
            torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
            self.assertEqual(controller._records, [])

    def test_eval_does_not_call_controller_and_ids_shapes_match(self):
        model = make_model().eval()
        positional, keywords = inputs(batch=1)
        with torch.no_grad(), mock.patch.object(ReplayBalancer, 'combine', side_effect=AssertionError('train only')):
            controller = ReplayBalancer(BalanceConfig(mode='adaptive'))
            model.get_loss(*positional, **keywords, replay_balancer=controller)
            triples, scores, relations = model.predict(positional[0], 2, positional[2], positional[1],
                keywords['model_output'], keywords['model_output_x_t'], keywords['tag'], False)
        self.assertEqual(triples.tolist(), [[0, 0, 1], [1, 2, 0]])
        self.assertEqual(tuple(scores.shape), (2, 6))
        self.assertEqual(tuple(relations.shape), (2, 4))

    def test_resume_needs_no_cross_stage_control_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = make_model()
            diffusion = copy.deepcopy(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            opt_diff = torch.optim.Adam(diffusion.parameters(), lr=.001)
            args = types.SimpleNamespace(run_dir=Path(tmp),
                resume=Path(tmp)/'latest.pt', _provenance={'protocol': 'official-dgar-v1', 'grb': 'adaptive'})
            positional, keywords = inputs()
            controller = ReplayBalancer(BalanceConfig(mode='adaptive'))
            controller.start_stage(2)
            loss, _, _ = run_loss(model, positional, keywords, rng_state('cpu'), controller)
            optimizer.step()
            optimizer.zero_grad()
            save_training_state(args, 2, model, diffusion, optimizer, opt_diff, 'cpu', None)
            state = rng_state('cpu')
            controller.start_stage(3)
            expected = run_loss(model, positional, keywords, state, controller)
            restored = make_model()
            restored_diff = copy.deepcopy(diffusion)
            restored_opt = torch.optim.Adam(restored.parameters(), lr=.001)
            restored_opt_diff = torch.optim.Adam(restored_diff.parameters(), lr=.001)
            stage, auxiliary = restore_training_state(args, restored, restored_diff,
                copy.deepcopy(restored), copy.deepcopy(restored_diff), restored_opt, restored_opt_diff, 'cpu')
            self.assertEqual(stage, 3)
            self.assertIsNone(auxiliary)
            resumed = ReplayBalancer(BalanceConfig(mode='adaptive'))
            resumed.start_stage(stage)
            actual = run_loss(restored, positional, keywords, rng_state('cpu'), resumed)
            for a, b in zip(actual[0], expected[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(resumed._moments, controller._moments, rtol=0, atol=0)
            self.assertEqual(restored.state_dict().keys(), model.state_dict().keys())

    def test_real_training_caller_passes_active_controller_then_abort(self):
        import src.main as main
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8', '--n-bases', '2',
            '--n-layers', '1', '--num_blocks', '1', '--add-static-graph', 'false',
            '--grb-mode', 'adaptive', '--diffusion_steps', '4', '--noise_schedule', 'cosine']))
        data = types.SimpleNamespace(num_nodes=8, num_rels=2,
            train=np.array([[0, 0, 1, 0]]*10), valid=np.array([[0, 1, 2, 1]]*10),
            test=np.array([[0, 0, 3, 2]]*10))
        seen = []
        original = RecurrentRGCN.get_loss
        class BatchComplete(Exception):
            pass
        def generate(_self, _args, stage, *_positional, **_keywords):
            self.assertEqual(stage, 2)
            return torch.tensor([[0, 0, 3]]), torch.ones(1, 3, 8), torch.tensor([[0, 0, 1]])
        def stop_after_backward(model, *positional, **keywords):
            controller = keywords['replay_balancer']
            self.assertIsInstance(controller, ReplayBalancer)
            self.assertEqual(controller.stage, 2)
            losses = original(model, *positional, **keywords)
            self.assertEqual(len(controller._records), 1)
            (.7*losses[0]+.3*losses[1]+losses[2]).backward()
            self.assertTrue(torch.isfinite(model.dynamic_emb.grad).all())
            seen.append(True)
            raise BatchComplete()
        def restore_fixture(a, model, *_positional):
            # Main's official protocol reloads best weights before stage 2.
            # Provide initial synthetic weights, not a trained checkpoint.
            torch.save(model.state_dict(), a.run_dir/'model.pt')
            return 2, None
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as stdout, \
             contextlib.redirect_stderr(io.StringIO()):
            args.data_root, args.run_dir = Path(temp)/'data', Path(temp)/'run'
            args.resume = args.run_dir/'latest.pt'
            directory = args.data_root/args.dataset
            cache = directory/'history_snap_v2'
            cache.mkdir(parents=True)
            for name in ('train', 'valid', 'test', 'stat'):
                (directory/f'{name}.txt').write_text('synthetic fixture\n')
            for stage, row in enumerate(((0, 0, 1), (0, 1, 2), (0, 0, 3))):
                (cache/f'snap_{stage}.pkl').write_bytes(pickle.dumps({'0': {row}}))
            with mock.patch.object(main.utils, 'load_data', return_value=data), \
                 mock.patch.object(main, 'fitlog'), \
                 mock.patch.object(main, 'restore_training_state', side_effect=restore_fixture), \
                 mock.patch.object(main.Att_Diffuse_model, 'gereration_feature', generate), \
                 mock.patch.object(RecurrentRGCN, 'get_loss', stop_after_backward), \
                 mock.patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer step allowed')), \
                 self.assertRaises(BatchComplete):
                main.run_experiment(args)
            self.assertIn('GRB mode: adaptive', stdout.getvalue())
            self.assertFalse((args.run_dir/'latest.pt').exists())
        self.assertEqual(seen, [True])
        self.assertFalse(torch.cuda.is_initialized())

    def test_summary_uses_final_sections_and_rejects_mismatched_runs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root/'logs').mkdir()
            runs = {}
            for group in ('original', 'half', 'adaptive'):
                directory = root/'experiment_runs'/group
                directory.mkdir(parents=True)
                runs[group] = directory
                config = {'protocol': 'official-dgar-v1', 'upstream_commit': 'same',
                    'source_sha256': 'same', 'data_sha256': 'same', 'history_sha256': 'same',
                    'settings': {'seed': 1, 'grb_fixed_weight': .5,
                        'grb_mode': {'original': 'original', 'half': 'fixed', 'adaptive': 'adaptive'}[group]}}
                (directory/'config.json').write_text(json.dumps(config))
                section = ('100%|progress| Current MRR (filter_ent): 0.6\n'
                    'Average MRR (macro, time_idx=1..T-2): 0.5\n'
                    'Average Hits@1 (macro, time_idx=1..T-2): 0.4\n'
                    'Average Hits@10 (macro, time_idx=1..T-2): 0.7\n'
                    'MRR (filter_ent): 0.49\nHits (filter_ent) @ 1: 0.39\n'
                    'Hits (filter_ent) @ 3: 0.55\nHits (filter_ent) @ 10: 0.69\n')
                (root/'logs'/f'{group}.log').write_text('Namespace(gpu=0, seed=1)\n'
                    'Current MRR (filter_ent): 0.9\nFinal full-pipeline TEST\n'+section+
                    'Final full-pipeline VALIDATION (supplementary; after TEST)\n'+section)
                if group != 'original':
                    record = {'stage': 1, 'updates': [{'weight': .5 if group == 'half' else .7,
                              'valid_signal': int(group == 'adaptive'), 'cosine': -.1}]}
                    (directory/'grb-stages.jsonl').write_text(json.dumps(record)+'\n')
            rows, weights = summarize(runs, root/'summary')
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row['current_mrr'] == .6 for row in rows))
            self.assertEqual(weights['adaptive']['mean'], .7)
            self.assertEqual(weights['adaptive']['negative_cosine_fraction'], 1.)
            config['source_sha256'] = 'different'
            (runs['adaptive']/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                summarize(runs, root/'bad-summary')


if __name__ == '__main__':
    unittest.main()
