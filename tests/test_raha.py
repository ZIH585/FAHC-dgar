"""CPU checks of Original equivalence, live RAHA training, and passive logging."""
import contextlib
import copy
import csv
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.raha import HistoricalAdapter
from src.raha_diagnostics import EvaluationRecorder, adapter_statistics, record_training_stage
from src.rrgcn import RecurrentRGCN
from src.runtime import (rng_state, restore_rng, settings, save_training_state,
                         restore_training_state, prepare_run)


ROOT = Path(__file__).resolve().parents[1]


def baseline_class():
    reference = ROOT / 'tests/reference/baseline_rrgcn.py.txt'
    data = reference.read_bytes()
    metadata = json.loads((reference.parent / 'raha_baseline.json').read_text())
    assert hashlib.sha256(data).hexdigest() == metadata['sha256']
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == metadata['git_blob']
    namespace = {'__name__': 'raha_original_baseline'}
    exec(compile(data, str(reference), 'exec'), namespace)
    return namespace['RecurrentRGCN']


class RAHATests(unittest.TestCase):
    def args(self, mode='original'):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8',
            '--n-bases', '2', '--n-layers', '1', '--num_blocks', '1',
            '--add-static-graph', 'false', '--diffusion_steps', '4',
            '--noise_schedule', 'cosine', '--run-dir', '/tmp/raha-synthetic',
            '--raha-mode', mode, '--raha-rank', '2']))
        args.num_nodes, args.num_rels = 6, 2
        return args

    def model(self, args, cls=RecurrentRGCN):
        with contextlib.redirect_stdout(io.StringIO()):
            return cls('convtranse', 'uvrgcn', 6, 2,
                1 if args.add_static_graph else 0, 2 if args.add_static_graph else 0, 8, 2, 'sub',
                sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=args)

    def graph(self):
        return [build_sub_graph(6, 2, np.array([[0, 0, 1], [2, 1, 3]]), False, -1)]

    def assert_state_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)

    def test_off_matches_frozen_baseline_init_parameters_and_rng(self):
        args = self.args()
        state = rng_state('cpu')
        reference = self.model(args, baseline_class())
        expected_rng = torch.get_rng_state().clone()
        restore_rng(state, 'cpu')
        actual = self.model(args)
        self.assertIsNone(actual.raha)
        self.assert_state_equal(actual.state_dict(), reference.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(list(dict(actual.named_parameters())), list(dict(reference.named_parameters())))

    def test_off_loss_backward_buffers_and_prediction_match_baseline(self):
        args = self.args()
        baseline = self.model(args, baseline_class())
        model = self.model(args)
        model.load_state_dict(baseline.state_dict())
        tags = torch.tensor([[0, 0, 1], [5, 1, 2], [1, 2, 0]])
        features = torch.randn(3, 3, 8)
        state = rng_state('cpu')

        def loss_and_predict(m):
            m.train()
            losses = m.get_loss(self.graph(), tags[:1], None, False,
                model_output=features, model_output_x_t=features[:, -1],
                tag=tags, diffuc='add', output_reply_triple=tags)
            (.7 * losses[0] + .3 * losses[1] + losses[2]).backward()
            m.eval()
            prediction = m.predict(self.graph(), 2, None, tags[:1],
                                    features, features[:, -1], tags, False)
            return losses, prediction

        expected_loss, expected_prediction = loss_and_predict(baseline)
        expected_rng = torch.get_rng_state().clone()
        restore_rng(state, 'cpu')
        actual_loss, actual_prediction = loss_and_predict(model)
        for a, b in zip(actual_loss + actual_prediction, expected_loss + expected_prediction):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for (name, a), (_, b) in zip(model.named_parameters(), baseline.named_parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad, name)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        self.assert_state_equal(model.state_dict(), baseline.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_enabled_starts_at_identity_and_does_not_shift_backbone_or_generator_rng(self):
        state = rng_state('cpu')
        baseline = self.model(self.args())
        expected_rng = torch.get_rng_state().clone()
        tags = torch.tensor([[0, 0, 1], [1, 1, 5], [1, 2, 0]])
        features = torch.arange(72).float().reshape(3, 3, 8)
        expected_ids, expected_bank = baseline.select_entity(tags, features, False)
        for mode in ('shared', 'role'):
            restore_rng(state, 'cpu')
            model = self.model(self.args(mode))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            core = {k: v for k, v in model.state_dict().items() if not k.startswith('raha.')}
            self.assert_state_equal(core, baseline.state_dict())
            ids, bank = model.select_entity(tags, features, False)
            torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(bank, expected_bank, rtol=0, atol=0)

    def test_shared_and_role_initial_projections_match_but_are_independent(self):
        state = torch.get_rng_state().clone()
        shared, role = HistoricalAdapter(8, 2, 'shared'), HistoricalAdapter(8, 2, 'role')
        torch.testing.assert_close(shared.shared.down.weight, role.head.down.weight, rtol=0, atol=0)
        torch.testing.assert_close(role.head.down.weight, role.tail.down.weight, rtol=0, atol=0)
        self.assertNotEqual(role.head.down.weight.data_ptr(), role.tail.down.weight.data_ptr())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in role.parameters()),
                         sum(p.numel() for p in HistoricalAdapter(8, 4, 'shared').parameters()))

    def test_role_mapping_keeps_occurrence_weighting_and_inverse_ids(self):
        model = self.model(self.args('role'))
        with torch.no_grad():
            model.raha.head.down.weight.fill_(.1)
            model.raha.head.up.weight.fill_(.2)
            model.raha.tail.down.weight.fill_(.3)
            model.raha.tail.up.weight.fill_(-.1)
        tags = torch.tensor([[0, 0, 1], [0, 1, 5], [1, 2, 0]])
        features = torch.randn(3, 3, 8)
        h, t = model.raha(features[:, 0], features[:, 2])
        ids, bank = model.select_entity(tags, features, False)
        torch.testing.assert_close(bank[0], (h[0] + h[1] + t[2]) / 3)
        torch.testing.assert_close(bank[1], (t[0] + h[2]) / 2)
        torch.testing.assert_close(bank[5], t[1])
        self.assertEqual(ids.tolist(), [0, 1, 5])
        self.assertEqual(float(bank[4].abs().sum()), 0)
        self.assertEqual(tags[2, 1].item(), 2)  # inverse relation is untouched

    def test_real_loss_updates_adapters_and_retains_finite_shapes(self):
        for mode in ('shared', 'role'):
            with self.subTest(mode=mode):
                model = self.model(self.args(mode))
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                tags = torch.tensor([[0, 0, 1], [2, 1, 5]])
                features = torch.randn(2, 3, 8)
                for step in range(2):
                    model.train()
                    e, r, static, _, _ = model.get_loss(self.graph(), tags[:1], None, False,
                        model_output=features, model_output_x_t=features[:, -1],
                        tag=tags, diffuc='add', output_reply_triple=tags)
                    loss = .7 * e + .3 * r + static
                    self.assertTrue(torch.isfinite(loss).all())
                    loss.backward()
                    for name, parameter in model.raha.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        if 'up' in name or step:
                            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
                    optimizer.step()
                    optimizer.zero_grad()
                self.assertTrue(all(float(p.norm()) > 0 for p in model.raha.parameters()))
                model.eval()
                expanded, scores, rel_scores = model.predict(self.graph(), 2, None,
                    tags[:1], features, features[:, -1], tags, False)
                self.assertEqual(tuple(expanded.shape), (2, 3))
                self.assertEqual(tuple(scores.shape), (2, 6))
                self.assertEqual(tuple(rel_scores.shape), (2, 4))
                self.assertTrue(torch.isfinite(scores).all())

    def test_empty_replay_no_old_data_and_singleton(self):
        model = self.model(self.args('role'))
        model.eval()
        tags = torch.tensor([[5, 1, 0]])
        for features in (None, torch.empty(0, 3, 8), torch.randn(1, 3, 8)):
            replay_tags = tags if features is not None and len(features) else torch.empty(0, 3, dtype=torch.long)
            tail = None if features is None else features[:, -1]
            expanded, scores, _ = model.predict(self.graph(), 2, None,
                tags, features, tail, replay_tags, False)
            self.assertEqual(tuple(scores.shape), (2, 6))
            self.assertTrue(torch.isfinite(scores).all())
        ids, bank = model.select_entity(torch.empty(0, 3, dtype=torch.long), torch.empty(0, 3, 8), False)
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(tuple(bank.shape), (6, 8))
        self.assertEqual(float(bank.sum()), 0)
        # The first trainable stage has no replay; adapters must not change.
        before = copy.deepcopy(model.raha.state_dict())
        model.train()
        e, r, _, _, _ = model.get_loss(self.graph(), tags, None, False, diffuc='add')
        (.7 * e + .3 * r).backward()
        self.assertTrue(all(p.grad is None for p in model.raha.parameters()))
        self.assert_state_equal(before, model.raha.state_dict())

    def test_double_precision_empty_adapter_and_invalid_shapes(self):
        adapter = HistoricalAdapter(8, 2).double()
        h, t = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
        torch.testing.assert_close(adapter(h, t)[0], h, rtol=0, atol=0)
        self.assertEqual(adapter(h[:0], t[:0])[0].shape, (0, 8))
        for a, b in ((h[:, :7], t), (h, t.float()), (h.long(), t.long())):
            with self.assertRaises(ValueError):
                adapter(a, b)
        with self.assertRaises(ValueError):
            HistoricalAdapter(8, 9)

    def test_guidance_uses_frozen_adapter_without_parameter_gradients(self):
        args = self.args('role')
        model = self.model(args).requires_grad_(False).train()
        with contextlib.redirect_stdout(io.StringIO()):
            generator = Att_Diffuse_model(create_model_diffu(args), args, 'uvrgcn',
                6, 2, max_time=4, dropout=.2).requires_grad_(False).train()
        tags = torch.tensor([[0, 0, 1]])
        calls = []
        handle = model.raha.register_forward_hook(lambda *a: calls.append(1))
        with torch.no_grad():
            out = generator(None, tags, args, False, False, ct=2,
                model=model, history_glist=self.graph())[1]
        handle.remove()
        self.assertTrue(calls)
        self.assertEqual(tuple(out.shape), (1, 3, 8))
        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNone(out.grad_fn)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(out.requires_grad)

    def test_default_static_branch_backward_and_prediction(self):
        args = self.args('role')
        args.add_static_graph = True
        with contextlib.redirect_stdout(io.StringIO()):
            model = RecurrentRGCN('convtranse', 'uvrgcn', 6, 2, 1, 2, 8, 2, 'sub',
                sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, use_static=True,
                weight=.5, discount=1, angle=10, args=args)
        static = build_sub_graph(8, 1, np.array([[0, 0, 6], [1, 0, 7]]), False, -1)
        tags, features = torch.tensor([[0, 0, 1]]), torch.randn(1, 3, 8)
        e, r, s, _, _ = model.get_loss(self.graph(), tags, static, False,
            model_output=features, model_output_x_t=features[:, -1],
            tag=tags, diffuc='add', output_reply_triple=tags)
        loss = .7 * e + .3 * r + s
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(torch.isfinite(model.raha.head.up.weight.grad).all())
        model.eval()
        _, scores, _ = model.predict(self.graph(), 2, static, tags,
            features, features[:, -1], tags, False)
        self.assertEqual(scores.shape, (2, 6))
        self.assertTrue(torch.isfinite(scores).all())

    def test_checkpoint_restores_adapter_and_adam_state(self):
        args = self.args('role')
        model = self.model(args)
        diffusion = torch.nn.Linear(2, 2)
        optimizer, diffusion_optimizer = torch.optim.Adam(model.parameters()), torch.optim.Adam(diffusion.parameters())
        # Real adapter update, then exercise production save/restore functions.
        h, t = torch.randn(2, 8), torch.randn(2, 8)
        sum(x.square().mean() for x in model.raha(h, t)).backward()
        optimizer.step()
        optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as tmp:
            args.run_dir = Path(tmp)
            args._provenance = {'protocol': 'raha-role-v1', 'synthetic': True}
            save_training_state(args, 2, model, diffusion, optimizer, diffusion_optimizer, 'cpu', None)
            restored = self.model(args)
            teacher = self.model(args).requires_grad_(False)
            new_diff, diff_teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
            new_optimizer = torch.optim.Adam(restored.parameters())
            new_diff_optimizer = torch.optim.Adam(new_diff.parameters())
            args.resume = args.run_dir / 'latest.pt'
            stage, _ = restore_training_state(args, restored, new_diff, teacher, diff_teacher,
                new_optimizer, new_diff_optimizer, 'cpu')
            self.assertEqual(stage, 3)
            self.assert_state_equal(restored.state_dict(), model.state_dict())
            self.assert_state_equal(teacher.state_dict(), model.state_dict())
            for name, parameter in restored.raha.named_parameters():
                old = dict(model.raha.named_parameters())[name]
                self.assert_state_equal(new_optimizer.state[parameter], optimizer.state[old])

    def test_cli_defaults_alias_validation_and_settings(self):
        parser = build_arg_parser()
        self.assertEqual(validate_args(parser.parse_args([])).raha_mode, 'original')
        self.assertEqual(validate_args(parser.parse_args(['--raha', '--run-dir', '/tmp/raha'])).raha_mode, 'role')
        for tokens in (['--raha'], ['--raha', '--run-dir', '/tmp/raha', '--raha-rank', '0'],
                       ['--raha', '--run-dir', '/tmp/raha', '--diffuc', 'false'],
                       ['--raha', '--run-dir', '/tmp/raha', '--delete_feature_reply', 'true'],
                       ['--raha-diagnostics'], ['--raha', '--run-dir', '/tmp/raha', '--grid-search']):
            with self.assertRaises(ValueError):
                validate_args(parser.parse_args(tokens))
        args = self.args()
        without_new_fields = types.SimpleNamespace(**{k: v for k, v in vars(args).items() if not k.startswith('raha_')})
        self.assertEqual(settings(args), settings(without_new_fields))
        self.assertNotIn('raha_diagnostics', settings(self.args('role')))

    def test_provenance_separates_modes_and_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data/TOY'
            (data / 'history_snap_v2').mkdir(parents=True)
            for name in ('train', 'valid', 'test', 'stat'):
                (data / f'{name}.txt').write_text('toy\n')
            args = self.args('role')
            args.data_root, args.dataset, args.run_dir = root / 'data', 'TOY', root / 'run'
            prepare_run(args)
            self.assertEqual(args._provenance['protocol'], 'raha-role-v1')
            args.test, args.raha_mode = True, 'shared'
            with self.assertRaises(ValueError):
                prepare_run(args)

    def test_diagnostics_are_passive_causal_and_recompute_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args('role')
            args.run_dir = Path(tmp)
            args._provenance = {'protocol': 'raha-role-v1'}
            model = self.model(args)
            history = [np.array([[0, 0, 1]]), np.array([[2, 1, 3]]), np.array([[4, 0, 5]])]
            features = torch.randn(1, 3, 8)
            state, weights = rng_state('cpu'), copy.deepcopy(model.state_dict())
            recorder = EvaluationRecorder(args, 'validation', history, 2)
            tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3], [4, 0, 5]])
            recorder.record(1, tags, torch.tensor([1, 2, 4, 8]), torch.tensor([1, 2, 4, 8]), model, features)
            recorder.record(2, tags, torch.ones(4), torch.ones(4), model, features)
            record_training_stage(args, 2, model, features)
            with contextlib.redirect_stdout(io.StringIO()):
                recorder.finish(2)
            rows = list(csv.DictReader((recorder.directory / 'queries.csv').open()))
            self.assertEqual([int(r['seen_before']) for r in rows[:4]], [1, 1, 0, 0])
            self.assertEqual([int(r['seen_before']) for r in rows[4:]], [1, 1, 1, 0])
            summary = json.loads((recorder.directory / 'summary.json').read_text())
            self.assertAlmostEqual(summary['macro_mrr'], .46875)
            self.assertAlmostEqual(summary['pooled']['mrr'], .734375)
            self.assertEqual(summary['current']['mrr'], 1.)
            self.assertEqual(summary['groups']['forward']['queries'], 6)
            self.assert_state_equal(model.state_dict(), weights)
            self.assertTrue(torch.equal(torch.get_rng_state(), state['torch']))
            self.assertEqual(adapter_statistics(model, features)['head_residual_mean_norm'], 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            with self.assertRaises(ValueError):
                recorder.record(1, tags, torch.ones(4), torch.ones(4), model, features)

    def test_actual_final_evaluation_with_diagnostics_keeps_original_ranks_rng_and_buffers(self):
        import pickle
        from src import main
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args()
            args.run_dir, args.data_root, args.dataset = Path(tmp) / 'run', Path(tmp) / 'data', 'TOY'
            args._provenance = {'protocol': 'official-dgar-v1'}
            cache = args.data_root / args.dataset / 'history_snap_v2'
            cache.mkdir(parents=True)
            history = [np.array([[0, 0, i + 1]]) for i in range(3)]
            queries = [np.array([[0, 0, i + 1], [2, 1, i + 3]]) for i in range(3)]
            for stage, triple in enumerate(history):
                (cache / f'snap_{stage}.pkl').write_bytes(pickle.dumps({'0': {tuple(triple[0])}}))
            model = self.model(args)
            weights, state = copy.deepcopy(model.state_dict()), rng_state('cpu')
            with contextlib.redirect_stdout(io.StringIO()):
                answers = main.utils.load_all_answers_for_time_filter(queries, 2, 6, False)
                rel_answers = main.utils.load_all_answers_for_time_filter(queries, 2, 6, True)

            def generator(sequence, tags, *a, **kw):
                return None, torch.rand(len(tags), 3, 8), None, None, None, None, None, None, None

            def execute(diagnostics):
                args.raha_diagnostics = diagnostics
                with contextlib.redirect_stdout(io.StringIO()):
                    return main.test(args, model, history, queries, 2, 6, False, answers,
                        rel_answers, 'unused', None, 'valid', None, None, None, generator)

            normal = execute(False)
            expected_rng, expected_weights = rng_state('cpu'), copy.deepcopy(model.state_dict())
            restore_rng(state, 'cpu')
            model.load_state_dict(weights)
            observed = execute(True)
            for a, b in zip(normal[:4], observed[:4]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(normal[6:], observed[6:])
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng['torch']))
            self.assertEqual(rng_state('cpu')['python'], expected_rng['python'])
            self.assert_state_equal(model.state_dict(), expected_weights)
            self.assertEqual(len(list((args.run_dir / 'diagnostics').glob('test-*/summary.json'))), 1)

    def test_summary_script_audits_csv_and_rejects_mismatched_cohorts(self):
        spec = importlib.util.spec_from_file_location('raha_summary', ROOT / 'scripts/summarize_raha.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for mode in ('original', 'shared', 'role'):
                args = self.args(mode)
                args.run_dir = root / mode
                args.run_dir.mkdir()
                args._provenance = {'protocol': 'official-dgar-v1' if mode == 'original' else f'raha-{mode}-v1',
                                    'settings': settings(args), 'source_sha256': 'S', 'data_sha256': 'D', 'history_sha256': 'H'}
                (args.run_dir / 'config.json').write_text(json.dumps(args._provenance))
                model = self.model(args)
                history = [np.array([[0, 0, 1]]), np.array([[2, 1, 3]]), np.array([[4, 0, 5]])]
                tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3], [3, 3, 2]])
                for split in ('test', 'validation'):
                    recorder = EvaluationRecorder(args, split, history, 2)
                    for stage in (1, 2):
                        ranks = torch.tensor([1, 2, 4, 8])
                        recorder.record(stage, tags, ranks, ranks, model, None)
                    with contextlib.redirect_stdout(io.StringIO()):
                        recorder.finish(2)
                runs.append(args.run_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                module.summarize(runs, root / 'summary')
            rows = list(csv.DictReader((root / 'summary/metrics.csv').open()))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(float(row['macro_mrr']) == .46875 for row in rows))
            self.assertTrue(all(float(row['macro_gain_pp']) == 0. for row in rows))
            config_path = runs[-1] / 'config.json'
            config = json.loads(config_path.read_text())
            config['history_sha256'] = 'DIFFERENT'
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                module.summarize(runs, root / 'bad-summary')
            self.assertFalse((root / 'bad-summary').exists())


if __name__ == '__main__':
    unittest.main()
