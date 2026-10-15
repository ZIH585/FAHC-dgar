"""Synthetic CPU tests: baseline contract, live PGR calls, causality and resume."""
import contextlib
import copy
import hashlib
import io
import importlib.util
import inspect
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.pgr import (PGRRuntime, SharedAdapter, aggregate, bank_variance,
                     calibrate, geometry_from_gradients, isolated_predictor,
                     shrink_noise, checkpoint_auxiliary)
from src.pgr_diagnostics import EvaluationRecorder, record_training_stage
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng, settings, save_training_state, restore_training_state

ROOT = Path(__file__).resolve().parents[1]


class PGRTests(unittest.TestCase):
    def args(self, mode='original', static=False):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--seed', '1', '--n-hidden', '8', '--hidden_size', '8',
            '--n-bases', '2', '--n-layers', '1', '--num_blocks', '1',
            '--add-static-graph', str(static).lower(), '--diffusion_steps', '4',
            '--noise_schedule', 'cosine', '--run-dir', '/tmp/pgr-synthetic',
            '--pgr-mode', mode, '--pgr-adapter-rank', '2', '--pgr-risk-rank', '2',
            '--pgr-geometry-facts', '2']))
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

    def assert_state(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)

    def test_off_matches_baseline_init_loss_gradients_prediction_and_rng(self):
        reference = ROOT / 'tests/reference/pgr_baseline_rrgcn.py.txt'
        meta = json.loads((reference.parent / 'pgr_baseline.json').read_text())
        data = reference.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), meta['sha256'])
        namespace = {'__name__': 'pgr_original_reference'}
        exec(compile(data, str(reference), 'exec'), namespace)
        args = self.args()
        state = rng_state('cpu')
        reference_model = self.model(args, namespace['RecurrentRGCN'])
        expected_rng = torch.get_rng_state().clone()
        restore_rng(state, 'cpu')
        actual = self.model(args)
        self.assertIsNone(actual.pgr_adapter)
        self.assert_state(actual.state_dict(), reference_model.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
        features = torch.randn(3, 3, 8)
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
            scores = model.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)
            results.append((loss.detach(), scores[1], torch.get_rng_state().clone()))
        for x, y in zip(results[0], results[1]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assert_state(actual.state_dict(), reference_model.state_dict())
        for (a, p), (b, q) in zip(actual.named_parameters(), reference_model.named_parameters()):
            self.assertEqual(a, b)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.assertFalse(any(k.startswith('pgr_') for k in settings(args)))

    def test_adapter_identity_rng_and_live_gradients(self):
        state = torch.get_rng_state().clone()
        adapter = SharedAdapter(8, 2)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.randn(3, 8)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        self.assertGreater(float(adapter.down.weight.norm()), 0)
        adapter(x).square().sum().backward()
        self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
        self.assertEqual(float(adapter.down.weight.grad.norm()), 0)
        with torch.no_grad():
            adapter.up.weight.add_(adapter.up.weight.grad, alpha=-.001)
        adapter.zero_grad()
        adapter(x).square().sum().backward()
        self.assertGreater(float(adapter.down.weight.grad.norm()), 0)

    def test_aggregation_self_loop_singleton_empty_and_variance(self):
        tags = torch.tensor([[0, 0, 1], [1, 2, 1], [1, 1, 2]])
        features = torch.randn(3, 3, 8)
        model = self.model(self.args())
        expected = model.select_entity(tags, features, False)
        observed = aggregate(tags, features, 6)
        torch.testing.assert_close(observed[0], expected[0])
        torch.testing.assert_close(observed[1], expected[1])
        variance = bank_variance(tags, torch.ones(3, 1), 6)
        torch.testing.assert_close(variance, torch.tensor([0, 2/16, 1, 0, 0, 0]))
        for tags, features in ((tags[:1], features[:1]), (tags[:0], features[:0])):
            ids, bank = aggregate(tags, features, 6)
            self.assertEqual(bank.shape, (6, 8))
            self.assertTrue(torch.isfinite(bank).all())

    def test_projection_random_spectrum_equal_energy_and_zero_geometry(self):
        gradients = [torch.randn(6, 8), torch.randn(6, 8)]
        variance = torch.rand(6)
        state = torch.get_rng_state().clone()
        q, w, stats = geometry_from_gradients(gradients, variance, 2, 1, 'geometry', 1)
        qr, wr, _ = geometry_from_gradients(gradients, variance, 2, 1, 'geometry-random', 1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(w, wr)
        torch.testing.assert_close(q.T @ q, torch.eye(2), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(q, qr))
        projected = shrink_noise(torch.eye(8), q, w, 'geometry')
        isotropic = shrink_noise(torch.eye(8), q, w, 'geometry-isotropic')
        torch.testing.assert_close(projected.square().sum(), isotropic.square().sum())
        self.assertLess(float(projected.square().sum()), 8)
        q0, w0, _ = geometry_from_gradients([g * 0 for g in gradients], variance, 2, 1, 'geometry', 1)
        self.assertEqual(float(w0.sum()), 0)

    def test_train_only_calibration_restores_modes_rng_buffers_gradients_and_scratch(self):
        args = self.args('geometry')
        model = self.model(args).requires_grad_(False).train()
        model.decoder_ob.bn0.eval()  # Also restore intentionally mixed modes.
        modes = [m.training for m in model.modules()]
        weights = copy.deepcopy(model.state_dict())
        old_h = model.h
        state = rng_state('cpu')
        tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
        result = calibrate(model, tags, torch.randn(3, 3, 8), torch.ones(3, 1),
            self.history(), tags[:1], None, args, 2)
        self.assertEqual(result['basis'].shape, (8, 2))
        self.assertGreater(result['stats']['trace'], 0)
        self.assertTrue(all(g > 0 for g in result['stats']['gradient_norms']))
        self.assert_state(weights, model.state_dict())
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertIs(model.h, old_h)
        self.assertNotIn('select_entity', model.__dict__)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(result['basis'].requires_grad)
        before = rng_state('cpu')
        with self.assertRaisesRegex(ValueError, 'deliberate'):
            with isolated_predictor(model):
                model.h = torch.randn(2)
                model.select_entity = None
                raise ValueError('deliberate')
        self.assertTrue(torch.equal(torch.get_rng_state(), before['torch']))
        self.assertIs(model.h, old_h)

    def test_actual_guided_sampler_stage_cache_gradients_and_resume(self):
        for mode in ('geometry', 'geometry-random', 'geometry-isotropic', 'mean', 'mean-shared', 'shared'):
            args = self.args(mode)
            args._pgr_runtime = runtime = PGRRuntime(args)
            model = self.model(args).requires_grad_(False).train()
            with contextlib.redirect_stdout(io.StringIO()):
                generator = Att_Diffuse_model(create_model_diffu(args), args,
                    'uvrgcn', 6, 2, max_time=4, dropout=.2).requires_grad_(False).train()
            tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
            runtime.begin_fit(1, self.history(), tags, None)
            runtime.end_fit()  # Original's first stage has no replay.
            runtime.begin_fit(2, self.history(), tags[:1], None)
            with torch.no_grad():
                result = generator(None, tags, args, False, False, ct=2,
                    model=model, history_glist=self.history())[1].detach()
            runtime.end_fit()
            self.assertEqual(result.shape, (3, 3, 8))
            self.assertTrue(torch.isfinite(result).all())
            new = PGRRuntime(args)
            new.load_state_dict(runtime.state_dict(), 2)
            args._pgr_runtime = new
            with torch.no_grad():
                repeated = generator(None, tags[:1], args, False, False, ct=2,
                    model=model, history_glist=self.history())[1]
            self.assertEqual(repeated.shape, (1, 3, 8))
            learner = self.model(args).train()
            learner.load_state_dict(model.state_dict())
            losses = learner.get_loss(self.history(), tags[:1], None, False,
                model_output=result, model_output_x_t=result[:, 2], tag=tags,
                diffuc='add', output_reply_triple=tags)
            loss = .7 * losses[0] + .3 * losses[1] + losses[2]
            loss.backward()
            if learner.pgr_adapter is not None:
                self.assertGreater(float(learner.pgr_adapter.up.weight.grad.norm()), 0)
            if mode.startswith('geometry'):
                with self.assertRaisesRegex(ValueError, 'No training-only'):
                    with new.at_stage(3):
                        new.sample(model, tags, result, result[:, 2], torch.ones(3, 1), result)
            args.vl = True
            with torch.no_grad():
                vl = generator(None, tags, args, False, False,
                    model=model, history_glist=self.history())[1]
            self.assertEqual(vl.shape, (3, 8))  # Diffusion validation is untouched.

    def test_static_graph_single_fact_calibration_and_private_sampling(self):
        args = self.args('geometry', static=True)
        model = self.model(args).requires_grad_(False)
        static = build_sub_graph(8, 1, np.array([[0, 0, 6], [2, 0, 7]]), False, -1)
        tags = torch.tensor([[0, 0, 1]])
        features = torch.randn(1, 3, 8)
        before = rng_state('cpu')
        result = calibrate(model, tags, features, torch.ones(1, 1), self.history(),
            torch.tensor([[0, 0, 1], [2, 1, 3], [1, 0, 3]]), static, args, 2)
        self.assertTrue(torch.equal(torch.get_rng_state(), before['torch']))
        self.assertEqual(result['stats']['train_current'], 2)
        self.assertEqual(result['stats']['train_past'], 1)
        self.assertEqual(result['basis'].device.type, 'cpu')

    def test_real_main_loop_calls_stage_hooks_and_saves_catalog(self):
        # Reuse the existing scalar/Adam harness, but execute this branch's REAL
        # loop with PGR enabled. Neural geometry is tested separately above.
        import test_official_protocol as harness
        source = inspect.getsource(harness.run_loop)
        source = source.replace("    args._provenance =", "    args.pgr_mode = 'geometry'\n"
            "    args._pgr_runtime = PGRRuntime(args)\n    args._provenance =", 1)
        source = source.replace("              save_training_state=save)",
            "              save_training_state=save, checkpoint_auxiliary=checkpoint_auxiliary)")
        namespace = dict(vars(harness), PGRRuntime=PGRRuntime, checkpoint_auxiliary=checkpoint_auxiliary)
        exec(compile(source, '<pgr-scalar-main-harness>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            namespace['run_loop'](ROOT / 'src/main.py', Path(directory))
            latest = torch.load(Path(directory) / 'latest.pt', weights_only=False)
            self.assertEqual(list(latest['auxiliary']['pgr']['catalog']), [1, 2, 3])

    def test_frozen_controls_match_bank_norm_and_preserve_model_predictions(self):
        spec = importlib.util.spec_from_file_location('pgr_frozen', ROOT / 'scripts/diagnose_pgr.py')
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        args = self.args('shared')
        args._pgr_runtime = PGRRuntime(args)
        model = self.model(args).requires_grad_(False).eval()
        with torch.no_grad():
            model.pgr_adapter.up.weight.normal_(0, .1)
        tags = torch.tensor([[0, 0, 1], [2, 1, 3], [1, 2, 0]])
        with contextlib.redirect_stdout(io.StringIO()):
            generator = Att_Diffuse_model(create_model_diffu(args), args,
                'uvrgcn', 6, 2, max_time=4, dropout=.2).requires_grad_(False).train()
        weights = copy.deepcopy(model.state_dict())
        gen_buffers = {n: b.clone() for n, b in generator.named_buffers()}
        state = rng_state('cpu')
        with diagnostic.stage_state(model, generator, torch.device('cpu'), 100):
            features, capture = diagnostic.capture_once(generator, model, args, tags,
                self.history(), None, 2, False)
            mean = torch.cat((capture['x0'][:, :2], capture['mean'][:, None]), 1)
            self.assertEqual(capture['sigma'].shape, (3, 1))
            ids, controls, gains = diagnostic.mapping_controls(model, tags, features, mean, 100)
            for name in ('identity_norm', 'random_spectrum_norm'):
                torch.testing.assert_close(controls[name].norm(dim=1), controls['learned'].norm(dim=1))
            natural = model.predict(self.history(), 2, None, tags[:1], features, features[:, 2], tags, False)[1]
            _, precomputed = diagnostic.score_bank(model, ids, controls['learned'], tags, features,
                self.history(), None, tags[:1], False)
            torch.testing.assert_close(natural, precomputed, atol=2e-6, rtol=2e-5)
        self.assert_state(weights, model.state_dict())
        self.assert_state(gen_buffers, {n: b for n, b in generator.named_buffers()})
        self.assertTrue(torch.equal(torch.get_rng_state(), state['torch']))

    def test_real_final_evaluation_reads_catalog_and_logging_does_not_change_scores(self):
        from src.main import test as evaluate
        from rgcn.utils import load_all_answers_for_time_filter
        args = self.args('geometry')
        args._pgr_runtime = runtime = PGRRuntime(args)
        model = self.model(args).requires_grad_(False)
        with contextlib.redirect_stdout(io.StringIO()):
            generator = Att_Diffuse_model(create_model_diffu(args), args,
                'uvrgcn', 6, 2, max_time=4, dropout=.2).requires_grad_(False).train()
        train = [np.array([[0, 0, 1], [2, 1, 3]])] * 3
        tags = torch.as_tensor(train[0])
        runtime.begin_fit(1, self.history(), tags, None)
        runtime.end_fit()
        runtime.begin_fit(2, self.history(), tags, None)
        with torch.no_grad():
            generator(None, tags, args, False, False, ct=2, model=model, history_glist=self.history())
        runtime.end_fit()
        answers = load_all_answers_for_time_filter(train, 2, 6, False)
        relation_answers = load_all_answers_for_time_filter(train, 2, 6, True)
        with tempfile.TemporaryDirectory() as directory:
            args.data_root, args.dataset = Path(directory), 'SMALL'
            args.run_dir = Path(directory) / 'run'
            args._provenance = {'protocol': 'synthetic-pgr'}
            cache = Path(directory) / 'SMALL/history_snap_v2'
            cache.mkdir(parents=True)
            for stage in range(3):
                with (cache / f'snap_{stage}.pkl').open('wb') as stream:
                    pickle.dump({'0': {(0, 0, 1)}, '2': {(2, 1, 3)}}, stream)
            state = rng_state('cpu')
            model_state, generator_state = copy.deepcopy(model.state_dict()), copy.deepcopy(generator.state_dict())
            results = []
            for diagnostic in (False, True):
                restore_rng(state, 'cpu')
                model.load_state_dict(model_state)
                generator.load_state_dict(generator_state)
                args.pgr_diagnostics = diagnostic
                with contextlib.redirect_stdout(io.StringIO()):
                    result = evaluate(args, model, train, train, 2, 6, False,
                        answers, relation_answers, None, None, 'train',
                        self.history(), list(tags), None, generator, phase='validation')
                results.append((result[:4], torch.get_rng_state().clone(), copy.deepcopy(generator.state_dict())))
            for x, y in zip(results[0][0], results[1][0]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertTrue(torch.equal(results[0][1], results[1][1]))
            self.assert_state(results[0][2], results[1][2])
            summary = next((args.run_dir / 'diagnostics').glob('validation-*/summary.json'))
            self.assertEqual(json.loads(summary.read_text())['pooled']['queries'], 8)

    def test_catalog_rejects_future_missing_corrupt_and_has_no_graphs(self):
        args = self.args('geometry')
        runtime = PGRRuntime(args)
        with self.assertRaises(ValueError):
            runtime.begin_fit(2, [], torch.empty(0, 3), None)
        runtime.begin_fit(1, [], torch.empty(0, 3), None)
        runtime.end_fit()
        self.assertIsNone(runtime.context)
        self.assertEqual(runtime.catalog[1]['weights'].numel(), 0)
        for completed in (0, 2):
            with self.assertRaises(ValueError):
                runtime.load_state_dict(runtime.state_dict(), completed)
        corrupt = runtime.state_dict()
        corrupt['catalog'][1]['basis'] = torch.ones(9, 0)
        with self.assertRaises(ValueError):
            runtime.load_state_dict(corrupt, 1)

    def test_pgr_checkpoint_restores_auxiliary_model_and_adam(self):
        args = self.args('geometry')
        with tempfile.TemporaryDirectory() as directory:
            args.run_dir = Path(directory)
            args._provenance = {'protocol': 'synthetic-pgr'}
            args._pgr_runtime = PGRRuntime(args)
            args._pgr_runtime.begin_fit(1, [], torch.empty(0, 3), None)
            args._pgr_runtime.end_fit()
            model = self.model(args)
            diffusion = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters())
            opt_diff = torch.optim.Adam(diffusion.parameters())
            save_training_state(args, 1, model, diffusion, optimizer, opt_diff, 'cpu', checkpoint_auxiliary(args))
            args.resume = args.run_dir / 'latest.pt'
            first, auxiliary = restore_training_state(args, model, diffusion,
                self.model(args), torch.nn.Linear(2, 2), optimizer, opt_diff, 'cpu')
            restored = PGRRuntime(args)
            restored.load_state_dict(auxiliary['pgr'], first - 1)
            self.assertEqual(list(restored.catalog), [1])

    def test_observer_is_passive_and_reports_both_directions(self):
        args = self.args('shared')
        model = self.model(args)
        with tempfile.TemporaryDirectory() as directory:
            args.run_dir = Path(directory)
            args._provenance = {'protocol': 'synthetic-pgr'}
            recorder = EvaluationRecorder(args, 'validation', [np.array([[0, 0, 1]])] * 3, 2)
            state = torch.get_rng_state().clone()
            for stage in (1, 2):
                recorder.record(stage, torch.tensor([[0, 0, 1], [1, 2, 0]]),
                    torch.tensor([2, 3]), torch.tensor([1, 2]), model, torch.randn(1, 3, 8))
            before = torch.get_rng_state().clone()
            recorder.finish(2)
            record_training_stage(args, 2, model, None)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            result = json.loads((recorder.directory / 'summary.json').read_text())
            self.assertEqual(result['groups']['inverse']['queries'], 2)
            self.assertEqual(result['macro_mrr'], .75)


if __name__ == '__main__':
    unittest.main()
