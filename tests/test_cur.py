"""Synthetic CPU tests for causal labels, real dual losses and routed scoring."""
import contextlib
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.cur import (CURRuntime, DUAL, SharedAdapter, checkpoint_auxiliary, chunks,
                     dual_loss, isolated_predictor, mean_sample, paired_observations,
                     query_features, reciprocal_rank, representations)
from src.cur_diagnostics import EvaluationRecorder, summarize_rows
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng, settings, save_training_state, restore_training_state

ROOT = Path(__file__).resolve().parents[1]


class CURTests(unittest.TestCase):
    def args(self, mode='utility', static=False):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--seed', '1', '--n-hidden', '8', '--hidden_size', '8',
            '--n-bases', '2', '--n-layers', '1', '--num_blocks', '1',
            '--add-static-graph', str(static).lower(), '--diffusion_steps', '4',
            '--noise_schedule', 'cosine', '--run-dir', '/tmp/cur-synthetic',
            '--cur-mode', mode, '--cur-adapter-rank', '2', '--cur-probe-facts', '3',
            '--cur-score-batch-size', '2', '--cur-router-steps', '2']))
        args.num_nodes, args.num_rels = 6, 2
        return args

    def model(self, args, cls=RecurrentRGCN):
        with contextlib.redirect_stdout(io.StringIO()):
            return cls('convtranse', 'uvrgcn', 6, 2,
                1 if args.add_static_graph else 0, 2 if args.add_static_graph else 0,
                8, 2, 'sub', sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=args)

    def history(self):
        return [build_sub_graph(6, 2, np.array([[0, 0, 1], [2, 1, 3]]), False, -1)]

    def inputs(self):
        tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
        features = torch.randn(3, 3, 8)
        return tags, features

    def assert_state(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)

    def test_off_is_exact_baseline_loss_gradients_prediction_and_rng(self):
        reference = ROOT / 'tests/reference/cur_baseline_rrgcn.py.txt'
        self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(), json.loads(
            (reference.parent / 'cur_baseline.json').read_text())['sha256'])
        namespace = {'__name__': 'cur_reference'}
        exec(compile(reference.read_bytes(), str(reference), 'exec'), namespace)
        args = self.args('original')
        state = rng_state('cpu')
        reference_model = self.model(args, namespace['RecurrentRGCN'])
        restore_rng(state, 'cpu')
        actual = self.model(args)
        self.assert_state(reference_model.state_dict(), actual.state_dict())
        tags, features = self.inputs()
        state = rng_state('cpu')
        results = []
        for model in (reference_model, actual):
            restore_rng(state, 'cpu')
            losses = model.get_loss(self.history(), tags[:1], None, False,
                model_output=features, model_output_x_t=features[:, 2], tag=tags,
                diffuc='add', output_reply_triple=tags)
            loss = .7 * losses[0] + .3 * losses[1] + losses[2]
            loss.backward()
            model.eval()
            scores = model.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)[1]
            results.append((loss.detach(), scores, torch.get_rng_state().clone()))
        for a, b in zip(results[0], results[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assert_state(reference_model.state_dict(), actual.state_dict())
        for (name, p), (_, q) in zip(reference_model.named_parameters(), actual.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
        self.assertFalse(any(key.startswith('cur_') for key in settings(args)))

    def test_adapter_identity_and_first_step_gradients(self):
        state = torch.get_rng_state().clone()
        adapter = SharedAdapter(8, 2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.randn(3, 8)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        adapter(x).square().sum().backward()
        self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
        self.assertEqual(float(adapter.down.weight.grad.norm()), 0)

    def test_true_dual_backward_trains_backbone_adapter_and_keeps_router_separate(self):
        model = self.model(self.args())
        tags, features = self.inputs()
        losses = dual_loss(model, self.history(), tags[:1], None, False,
            model_output=features, model_output_x_t=features[:, 2], tag=tags,
            diffuc='add', output_reply_triple=tags)
        loss = .7 * losses[0] + .3 * losses[1] + losses[2]
        loss.backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(float(model.cur_adapter.up.weight.grad.norm()), 0)
        self.assertGreater(float(model.dynamic_emb.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.cur_router.parameters()))

    def test_no_injection_loss_still_uses_past_labels_and_not_feature_values(self):
        model = self.model(self.args()).eval()
        tags, features = self.inputs()
        def loss(f, past):
            return model.get_loss(self.history(), tags[:1], None, False,
                model_output=f, model_output_x_t=f[:, 2], tag=tags,
                diffuc='add', output_reply_triple=past,
                feature_injection=False, entity_only=True)[0]
        a = loss(features, tags)
        b = loss(features * 100, tags)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = tags.clone()
        changed[:, 2] = (changed[:, 2] + 2) % 6
        self.assertNotEqual(float(a), float(loss(features, changed)))

    def test_router_inputs_and_decisions_are_independent_of_targets(self):
        model = self.model(self.args()).eval()
        tags, features = self.inputs()
        queries = model.get_all_triple(tags[:1])
        with torch.no_grad():
            a = paired_observations(model, self.history(), None, queries, features, tags, True)
            changed = queries.clone()
            changed[:, 2] = (changed[:, 2] + 1) % 6
            b = paired_observations(model, self.history(), None, changed, features, tags, True)
        for key in ('features', 'probability', 'use_replay', 'scores'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            query_features(torch.zeros(2, 6), torch.zeros(2, 6), queries, None, 2)

    def test_prediction_uses_router_and_survives_weights_reload(self):
        model = self.model(self.args()).eval()
        tags, features = self.inputs()
        queries = model.get_all_triple(tags[:1])
        with torch.no_grad():
            pair = representations(model, self.history(), None, features, tags)
            expected = [model.decoder_ob(e, r, queries, mode='test') for e, r in pair]
            for bias, index in ((30, 0), (-30, 1)):
                model.cur_router.network[-1].bias.fill_(bias)
                output = model.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)
                torch.testing.assert_close(output[1], expected[index], rtol=0, atol=0)
            reloaded = self.model(self.args()).eval()
            reloaded.load_state_dict(model.state_dict(), strict=True)
            restored = reloaded.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)
            torch.testing.assert_close(restored[1], output[1], rtol=0, atol=0)

    def test_router_fit_is_preupdate_detached_and_preserves_teacher_and_rng(self):
        args = self.args()
        args.cur_probe_facts = 10
        model = self.model(args)
        teacher = copy.deepcopy(model).requires_grad_(False).train()
        teacher.decoder_ob.eval()  # Mixed original modes must be restored.
        before = copy.deepcopy(teacher.state_dict())
        modes = [m.training for m in teacher.modules()]
        tags, features = self.inputs()
        state = rng_state('cpu')
        runtime = CURRuntime(args, model)
        runtime.fit(1, teacher, self.history(), tags, None, features, tags)
        self.assert_state(before, teacher.state_dict())
        self.assertEqual(modes, [m.training for m in teacher.modules()])
        self.assertTrue(torch.equal(state['torch'], torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in model.cur_router.parameters()))
        self.assertEqual(runtime.stats['train']['queries'], 6)
        self.assertFalse(any(torch.is_tensor(v) for v in runtime.stats.values()))
        runtime.complete(1)
        with self.assertRaises(ValueError):
            runtime.fit(3, teacher, self.history(), tags, None, features, tags)

    def test_empty_replay_first_stage_and_small_candidate_pool(self):
        args = self.args()
        model = self.model(args)
        runtime = CURRuntime(args, model)
        tags, features = self.inputs()
        runtime.fit(1, model, self.history(), tags[:1], None, None, None)
        runtime.complete(1)
        for f in (None, features[:0]):
            output = model.eval().predict(self.history(), 2, None, tags[:1], f, None, tags, False)
            self.assertEqual(output[1].shape, (2, 6))
        runtime.fit(2, copy.deepcopy(model).requires_grad_(False), self.history(), tags[:1], None, features[:1], tags[:1])
        self.assertEqual(runtime.stats['train']['queries'], 2)
        self.assertIsNone(runtime.stats['holdout'])
        self.assertEqual([(s.start, s.stop) for s in chunks(5, 2)], [(0, 2), (2, 5)])

    def test_actual_guided_mean_sampler_and_dual_loss_cpu_offload(self):
        args = self.args()
        model = self.model(args)
        teacher = copy.deepcopy(model).requires_grad_(False).train()
        tags, _ = self.inputs()
        with contextlib.redirect_stdout(io.StringIO()):
            generator = Att_Diffuse_model(create_model_diffu(args), args, 'uvrgcn', 6, 2, max_time=4, dropout=.2)
        generator.requires_grad_(False).train()
        with torch.autograd.graph.save_on_cpu(pin_memory=False), torch.no_grad():
            features = generator(None, tags, args, False, False, ct=2, model=teacher, history_glist=self.history())[1].detach()
        self.assertEqual(features.shape, (3, 3, 8))
        self.assertTrue(torch.isfinite(features).all())
        x0, mean, original = torch.randn(3, 3, 8), torch.randn(3, 8), torch.randn(3, 3, 8)
        torch.testing.assert_close(mean_sample(args, x0, mean, original)[:, 2], mean)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            losses = dual_loss(model, self.history(), tags[:1], None, False,
                model_output=features, model_output_x_t=features[:, 2], tag=tags,
                diffuc='add', output_reply_triple=tags)
            (.7 * losses[0] + .3 * losses[1] + losses[2]).backward()
        self.assertTrue(torch.isfinite(model.dynamic_emb.grad).all())

    def test_rank_utility_sign_and_ties(self):
        scores = torch.tensor([[3., 2., 1.], [1., 1., 1.]])
        rr = reciprocal_rank(scores, torch.tensor([1, 0]))
        torch.testing.assert_close(rr, torch.tensor([.5, .5]))

    def test_router_optimizer_and_prediction_checkpoint_roundtrip(self):
        args = self.args()
        model = self.model(args)
        runtime = CURRuntime(args, model)
        tags, features = self.inputs()
        runtime.fit(1, copy.deepcopy(model).requires_grad_(False), self.history(), tags, None, features, tags)
        runtime.complete(1)
        clone = self.model(args)
        clone.load_state_dict(model.state_dict())
        resumed = CURRuntime(args, clone)
        resumed.load_state_dict(runtime.state_dict(), 1)
        self.assertEqual(resumed.completed, 1)
        self.assertEqual(len(resumed.optimizer.state), len(runtime.optimizer.state))
        with self.assertRaises(ValueError):
            resumed.load_state_dict(runtime.state_dict(), 2)
        for engine in (runtime, resumed):
            engine.fit(2, copy.deepcopy(model).requires_grad_(False), self.history(), tags, None, features, tags)
        self.assert_state(model.cur_router.state_dict(), clone.cur_router.state_dict())

    def test_frozen_diagnostic_whole_flow_on_synthetic_cache(self):
        from scripts import diagnose_cur as diagnostic
        args = self.args()
        model = self.model(args).requires_grad_(False).eval()
        model.cur_router.requires_grad_(True)
        with contextlib.redirect_stdout(io.StringIO()):
            generator = Att_Diffuse_model(create_model_diffu(args), args, 'uvrgcn', 6, 2, max_time=3, dropout=.2).requires_grad_(False).train()
        before = copy.deepcopy(model.state_dict())
        train = [np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64) for _ in range(3)]
        valid = [np.array([[0, 0, 2]], dtype=np.int64) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            import pickle
            args.data_root = Path(directory)
            args.dataset = 'SMALL'
            cache = args.data_root / args.dataset / 'history_snap_v2'
            cache.mkdir(parents=True)
            (cache / 'snap_0.pkl').write_bytes(pickle.dumps({'0': {(0, 0, 1)}}))
            output = Path(directory) / 'output'
            argv = ['diagnose_cur.py', '--gpu', '-1', '--run-dir', str(Path(directory) / 'source'),
                    '--output', str(output), '--stages', '2', '--facts', '1']
            payload = (args, {'synthetic': True}, train, valid, None, model, generator, torch.device('cpu'))
            with patch.object(diagnostic, 'load', return_value=payload), patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                diagnostic.main()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['test_queries_scored'], 0)
            self.assertEqual(summary['backbone_updates'], 0)
            self.assertEqual(summary['stages'][0]['stage'], 2)
            self.assertTrue((output / 'queries.csv').exists())
        for name, value in before.items():
            if not name.startswith('cur_router.'):
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_heldout_directions_stay_together_and_router_has_live_gradients(self):
        args = self.args()
        args.cur_probe_facts = 4
        model = self.model(args)
        runtime = CURRuntime(args, model)
        tags, features = self.inputs()
        current = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 3], [3, 1, 4]])
        observed = {'features': torch.randn(8, 10),
                    'rr_r': torch.tensor([1., .5, 1., .5, 1., .5, 1., .5]),
                    'rr_0': torch.tensor([.5, 1., .5, 1., .5, 1., .5, 1.])}
        old = copy.deepcopy(model.cur_router.state_dict())
        with patch('src.cur.paired_observations', return_value=observed):
            runtime.fit(1, model, self.history(), current, None, features, tags)
        self.assertEqual(runtime.stats['train']['queries'], 6)
        self.assertEqual(runtime.stats['holdout']['queries'], 2)
        self.assertGreater(len(runtime.optimizer.state), 0)
        self.assertTrue(any(not torch.equal(v, old[k]) for k, v in model.cur_router.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.cur_router.parameters()))

    def test_readonly_pgr_adapter_migration_is_strict_and_exact(self):
        from scripts.diagnose_cur import migrate_weights
        model = self.model(self.args())
        state = {k.replace('cur_adapter.', 'pgr_adapter.'): v.clone()
                 for k, v in model.state_dict().items() if not k.startswith('cur_router.')}
        clone = self.model(self.args())
        migrate_weights(clone, state)
        for key, value in model.state_dict().items():
            if not key.startswith('cur_router.'):
                torch.testing.assert_close(clone.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            migrate_weights(clone, dict(state, unknown_tensor=torch.ones(1)))

    def test_real_main_loop_calls_fit_dual_loss_and_saves_auxiliary(self):
        import test_official_protocol as harness
        events = []
        class Runtime:
            def fit(self, stage, *a): events.append(('fit', stage))
            def complete(self, stage): events.append(('complete', stage))
            def state_dict(self): return {'completed_stage': events[-1][1]}
        def dual(*a, **kw):
            events.append(('dual', 1))
            return dual_loss(*a, **kw)
        source = inspect.getsource(harness.run_loop)
        source = source.replace('    counts = {}', "    args.cur_mode = 'utility'\n    args._cur_runtime = Runtime()\n    counts = {}")
        source = source.replace('              save_training_state=save)',
            '              save_training_state=save, dual_loss=dual, checkpoint_auxiliary=checkpoint_auxiliary)')
        namespace = dict(vars(harness), Runtime=Runtime, dual=dual, checkpoint_auxiliary=checkpoint_auxiliary)
        exec(compile(source, '<cur-scalar-main-harness>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            namespace['run_loop'](ROOT / 'src/main.py', Path(directory))
            saved = torch.load(Path(directory) / 'latest.pt', weights_only=False)
        self.assertEqual([x[1] for x in events if x[0] == 'fit'], [1, 2, 3])
        self.assertEqual(sum(x[0] == 'dual' for x in events), 9)
        self.assertEqual(saved['auxiliary']['cur']['completed_stage'], 3)

    def test_passive_final_recorder_reports_paper_metrics_and_route_decisions(self):
        args = self.args()
        model = self.model(args).eval()
        tags, features = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            args.run_dir = Path(directory)
            args._provenance = {'synthetic': True}
            history = [np.array([[0, 0, 1]]) for _ in range(3)]
            recorder = EvaluationRecorder(args, 'validation', history, 2)
            for stage in (1, 2):
                triples, _, _ = model.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)
                recorder.record(stage, triples, torch.tensor([1, 2]), torch.tensor([1, 2]), model, features)
            recorder.finish(2)
            result = json.loads((recorder.directory / 'summary.json').read_text())
            self.assertEqual(result['macro_mrr'], .75)
            self.assertEqual(result['macro_hits1'], .5)
            self.assertEqual(result['macro_hits10'], 1)
            self.assertTrue(all(row['route_probability'] is not None for row in recorder.rows))


if __name__ == '__main__':
    unittest.main()
