"""CPU guards for BCR retrieval, chronology, real shapes and label blindness."""
import contextlib
import copy
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from src.bcr import BCR, HistoricalIndex, Replay, choose, isolated_generation, predict
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng
from src.runtime import PROTOCOL, UPSTREAM_COMMIT, fingerprint, settings
from rgcn.utils import build_sub_graph
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('calls', torch.tensor(0))

    def forward(self, sequence, tags, args, *a, **kw):
        self.calls.add_(len(tags))
        features = torch.randn(len(tags), 3, args.n_hidden) + self.calls
        return (None, features, None, None, None, None, None, None, None)


class BCRTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train = [np.array([[0, 0, 1], [2, 1, 0], [1, 0, 3]], dtype=np.int64),
                      np.array([[3, 1, 0], [0, 0, 2]], dtype=np.int64),
                      np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64),
                      np.array([[4, 0, 5]], dtype=np.int64)]
        cache = self.root / 'SMALL/history_snap_v2'
        cache.mkdir(parents=True)
        for i, snap in enumerate(self.train):
            d = {str(int(s)): {tuple(map(int, row)) for row in snap if row[0] == s}
                 for s in set(snap[:, 0])}
            (cache / f'snap_{i}.pkl').write_bytes(pickle.dumps(d))
        self.args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--dataset', 'SMALL', '--data-root', str(self.root),
            '--run-dir', str(self.root / 'run'), '--bcr', '--n-hidden', '8',
            '--hidden_size', '8', '--n-bases', '2', '--n-layers', '1',
            '--add-static-graph', 'false']))
        self.args.run_dir.mkdir()
        self.manager = BCR(self.args, self.train, 2, 6)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return RecurrentRGCN('convtranse', 'uvrgcn', 6, 2, 0, 0, 8, 2,
                'sub', sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=self.args)

    def test_training_only_incident_retrieval_and_small_pools(self):
        out, incident, all_facts = self.manager.index.pools([0], [0], 2, 1)
        self.assertEqual(out, {(0, 0, 1)})
        self.assertEqual(incident, {(0, 0, 1), (2, 1, 0)})
        self.assertEqual(len(all_facts), 3)
        self.assertEqual(choose(incident, 9, 1), sorted(incident))
        self.assertEqual(choose(incident, 0, 1), [])
        self.assertEqual(choose([], 8, 1), [])
        for index in [1, 2, -1]:
            with self.assertRaises(ValueError):
                self.manager.index.pools([0], [index], 2, 1)
        with self.assertRaises(ValueError):
            self.manager.index.pools([6], [0], 2, 1)
        bad = HistoricalIndex([np.array([[0, 2, 1]])], 2, 6)
        with self.assertRaises(ValueError):
            bad.snapshot(0, 1, 1)

    def test_time_ranges_match_original_and_zero_replay(self):
        self.assertEqual(self.manager.times(1), ([], 0))
        self.assertEqual(self.manager.times(2), ([0], 1))
        self.assertEqual(self.manager.times(2, final=True), ([0, 1], 2))
        gen, model = Generator(), self.model()
        triples = torch.as_tensor(self.train[1])
        self.assertEqual(self.manager.training(1, triples, gen, model, [], None, None, False)[1:], (None, None))
        self.assertEqual(int(gen.calls), 0)
        result = self.manager.evaluation(1, [0], [1], gen, model, [], None,
                                         None, 'cpu', False, seed=99)
        self.assertTrue(all(x.features is None and x.tags.shape == (0, 3) for x in result))

    def test_training_fact_budget_and_forward_ids(self):
        gen, model = Generator(), self.model()
        _, features, tags = self.manager.training(2, torch.as_tensor(self.train[2]),
            gen, model, [], None, None, False)
        self.assertEqual(len(tags), self.manager.legacy_budget([0, 2], [0]))
        self.assertEqual(features.shape, (2, 3, 8))
        self.assertFalse(features.requires_grad)
        self.assertTrue((tags[:, 1] < 2).all())
        self.assertEqual(int(gen.calls), 2)

    def test_cache_is_unchanged_and_optional_inverse_not_duplicated(self):
        p = self.root / 'SMALL/history_snap_v2/snap_0.pkl'
        original = p.read_bytes()
        self.assertEqual(self.manager.legacy_budget([0], [0]), 1)
        self.assertEqual(p.read_bytes(), original)
        d = pickle.loads(original)
        d['1'].add((1, 2, 0))
        p.write_bytes(pickle.dumps(d))
        saved = p.read_bytes()
        self.assertEqual(self.manager.legacy_budget([0, 1], [0]), 2)
        self.assertEqual(p.read_bytes(), saved)

    def test_directional_budget_and_labels_do_not_cross(self):
        gen, model = Generator().train(), self.model().eval()
        kwargs = dict(generator=gen, guider=model, history=[], last=None,
                      static=None, device='cpu', cuda=False, final=True, seed=43)
        a = self.manager.evaluation(2, [0], [1], **kwargs)
        b = self.manager.evaluation(2, [0], [2], **kwargs)
        c = self.manager.evaluation(2, [3], [1], **kwargs)
        torch.testing.assert_close(a[0].tags, b[0].tags, rtol=0, atol=0)
        torch.testing.assert_close(a[0].features, b[0].features, rtol=0, atol=0)
        torch.testing.assert_close(a[1].tags, c[1].tags, rtol=0, atol=0)
        torch.testing.assert_close(a[1].features, c[1].features, rtol=0, atol=0)
        self.assertEqual(len(a[0].tags), 2)  # outgoing count of entity 0 in t0,t1
        self.assertEqual(len(a[1].tags), 1)
        self.assertEqual(int(gen.calls), 0)

    def test_isolated_generation_restores_rng_buffers_on_failure(self):
        gen, model = Generator().train(), self.model().train()
        state = rng_state('cpu')
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with isolated_generation(gen, model, 'cpu', 3):
                gen.calls.add_(3)
                model.decoder_ob.bn0.running_mean.add_(10)
                torch.rand(2)
                raise RuntimeError('interrupted branch')
        self.assertTrue(gen.training and model.training)
        self.assertEqual(int(gen.calls), 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), state['torch']))
        for k, value in before.items():
            torch.testing.assert_close(model.state_dict()[k], value, rtol=0, atol=0)

    def test_real_backward_and_bidirectional_singleton_scores(self):
        model, gen = self.model(), Generator()
        graph = [build_sub_graph(6, 2, self.train[0], False, -1)]
        _, features, tags = self.manager.training(2, torch.as_tensor(self.train[2]),
            gen, model, graph, None, None, False)
        model.train()
        e, r, static, _, _ = model.get_loss(graph, torch.tensor([[0, 0, 1]]), None,
            False, model_output=features, model_output_x_t=features[:, -1],
            tag=tags, diffuc='add', output_reply_triple=tags)
        loss = .7 * e + .3 * r + static
        loss.backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(model.dynamic_emb.grad).all())
        model.eval()
        model.zero_grad(set_to_none=True)
        replay = self.manager.evaluation(2, [0], [1], gen, model, graph, None,
                                         None, 'cpu', False, final=True, seed=8)
        triples, scores, rscores = predict(model, graph, 2, None, torch.tensor([[0, 0, 1]]), replay, False)
        self.assertEqual(triples.tolist(), [[0, 0, 1], [1, 2, 0]])
        self.assertEqual(scores.shape, (2, 6))
        self.assertEqual(rscores.shape, (2, 4))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # Replacing the forward target changes only the inverse known query;
        # the forward entity score must be exactly unchanged.
        changed = self.manager.evaluation(2, [0], [2], gen, model, graph, None,
                                         None, 'cpu', False, final=True, seed=8)
        _, altered, _ = predict(model, graph, 2, None, torch.tensor([[0, 0, 2]]), changed, False)
        torch.testing.assert_close(scores[0], altered[0], rtol=0, atol=0)
        empty = Replay(torch.empty(0, 3, dtype=torch.long), None)
        _, scores, _ = predict(model, graph, 2, None, torch.tensor([[0, 0, 1]]), (empty, empty), False)
        self.assertTrue(torch.isfinite(scores).all())

    def test_rebuilt_state_reproduces_selection_features_and_rng(self):
        gen, model = Generator(), self.model()
        state = rng_state('cpu')
        a = self.manager.evaluation(2, [0], [1], gen, model, [], None, None, 'cpu', False, final=True)
        after = rng_state('cpu')
        restore_rng(state, 'cpu')
        rebuilt = BCR(self.args, self.train, 2, 6)
        b = rebuilt.evaluation(2, [0], [1], gen, model, [], None, None, 'cpu', False, final=True)
        for x, y in zip(a, b):
            torch.testing.assert_close(x.tags, y.tags, rtol=0, atol=0)
            torch.testing.assert_close(x.features, y.features, rtol=0, atol=0)
        self.assertEqual(random.getstate(), after['python'])
        self.assertTrue(torch.equal(torch.get_rng_state(), after['torch']))

    def test_actual_guided_generation_modes_buffers_and_no_graph_cache(self):
        self.args.diffusion_steps = 4
        self.args.noise_schedule = 'cosine'
        self.args.num_blocks = 1
        self.args.num_nodes, self.args.num_rels = 6, 2
        model = self.model().requires_grad_(False).train()
        with contextlib.redirect_stdout(io.StringIO()):
            gen = Att_Diffuse_model(create_model_diffu(self.args), self.args,
                'uvrgcn', 6, 2, max_time=4, dropout=.2).requires_grad_(False).train()
        graph = [build_sub_graph(6, 2, self.train[0], False, -1)]
        before = [{k: v.clone() for k, v in module.state_dict().items()} for module in (model, gen)]
        a = self.manager.evaluation(2, [0], [1], gen, model, graph, None,
                                     None, 'cpu', False, final=True, seed=18)
        b = self.manager.evaluation(2, [0], [2], gen, model, graph, None,
                                     None, 'cpu', False, final=True, seed=18)
        self.assertTrue(model.training and gen.training)
        torch.testing.assert_close(a[0].features, b[0].features, rtol=0, atol=0)
        for module, weights in zip((model, gen), before):
            for key, tensor in weights.items():
                torch.testing.assert_close(module.state_dict()[key], tensor, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        for replay in a:
            self.assertFalse(replay.features.requires_grad)
            self.assertIsNone(replay.features.grad_fn)

    def test_real_main_evaluation_calls_two_contexts(self):
        from src.main import test, test_single
        model, gen = self.model().eval(), Generator().train()
        self.args._bcr = self.manager
        valid = [np.array([[0, 0, 1]], dtype=np.int64)] * 4
        answers = [{0: {0: {1}}, 1: {2: {0}}}] * 4
        graph = [build_sub_graph(6, 2, self.train[0], False, -1)]
        replays = self.manager.evaluation(2, [0], [1], gen, model, graph,
            None, None, 'cpu', False, seed=1)
        with contextlib.redirect_stdout(io.StringIO()):
            score, _, rank = test_single(self.args, model, graph,
                torch.tensor([[0, 0, 1]]), 2, 2, 6, False, answers, None,
                bcr_replays=replays)
        self.assertEqual(len(rank), 2)
        self.assertTrue(torch.isfinite(score))
        rel_answers = [{0: {1: {0}}, 1: {0: {2}}}] * 4
        with mock.patch.object(self.manager, 'evaluation', wraps=self.manager.evaluation) as retrieve:
            with contextlib.redirect_stdout(io.StringIO()):
                result = test(self.args, model, self.train, valid, 2, 6,
                    False, answers, rel_answers, 'unused.pt', None, 'validation',
                    graph, None, None, gen)
            self.assertEqual(retrieve.call_count, 3)
        self.assertEqual(len(result[6]), 3)
        self.assertEqual(int(gen.calls), 0)

    def test_real_main_first_loss_uses_bcr_training_output(self):
        # Execute the actual chronological body only until its first loss.
        # No backward, optimizer step, epoch or real data loading.
        from test_official_protocol import code_from
        from rgcn.utils import EarlyStopping
        self.args._bcr = self.manager
        with contextlib.redirect_stdout(io.StringIO()):
            model, gen = self.model(), Generator()
        torch.save(model.state_dict(), self.args.run_dir / 'model.pt')
        observed = []

        class ReachedLoss(Exception):
            pass

        def capture(*a, **kw):
            observed.append(kw)
            raise ReachedLoss()

        ns = dict(args=self.args, model=model, model_repre=model, model_diffpre=gen,
            first_stage=2, train_list=self.train, use_cuda=False, run_device=torch.device('cpu'),
            num_nodes=6, num_rels=2, static_graph=None, static_graph_diffu=None,
            model_state_file=self.args.run_dir / 'model.pt', diffu_model_state_file=self.args.run_dir / 'diffusion.pt',
            torch=torch, np=np, tqdm=lambda x: x, EarlyStopping=EarlyStopping,
            build_sub_graph=build_sub_graph)
        with mock.patch.object(model, 'get_loss', side_effect=capture):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ReachedLoss):
                    exec(code_from(Path(__file__).resolve().parents[1] / 'src/main.py',
                        'run_experiment', loop_only=True), ns)
        self.assertEqual(len(observed), 1)
        kw = observed[0]
        self.assertEqual(kw['model_output'].shape, (2, 3, 8))
        self.assertEqual(kw['tag'].shape, (2, 3))
        self.assertFalse(kw['model_output'].requires_grad)

    def test_cli_guards_default_off_and_alias(self):
        p = build_arg_parser()
        self.assertEqual(validate_args(p.parse_args([])).bcr_mode, 'original')
        for argv in [['--bcr'], ['--bcr', '--run-dir', str(self.root), '--delete_his_prompt', 'true']]:
            with self.assertRaises(ValueError):
                validate_args(p.parse_args(argv))
        self.assertEqual(self.args.bcr_mode, 'incident')
        self.assertEqual(self.args._protocol, 'bcr-directional-v1')

    def test_diagnostic_end_to_end_toy_checkpoint_and_read_only_inputs(self):
        from scripts.diagnose_bcr import make_models, parser, run
        root = Path(__file__).resolve().parents[1]
        dataset = self.root / 'ICEWS14s'  # accepted loader name, synthetic temp files
        cache = dataset / 'history_snap_v2'
        cache.mkdir(parents=True)
        (dataset / 'entity2id.txt').write_text(''.join(f'e{i}\t{i}\n' for i in range(6)))
        (dataset / 'relation2id.txt').write_text('r0\t0\nr1\t1\n')
        (dataset / 'stat.txt').write_text('6\t2\n')
        pattern = [(0, 0, 1), (2, 1, 0), (1, 0, 3), (3, 1, 2)]
        for t, name in enumerate(('train', 'valid', 'test')):
            rows = [pattern[i % 4] for i in range(10)]
            (dataset / f'{name}.txt').write_text(''.join(f'{s}\t{r}\t{o}\t{t+1}\n' for s, r, o in rows))
            d = {}
            for s, r, o in rows[:8]:
                d.setdefault(str(s), set()).add((s, r, o))
            (cache / f'snap_{t}.pkl').write_bytes(pickle.dumps(d))
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--dataset', 'ICEWS14s', '--data-root', str(self.root),
            '--n-hidden', '8', '--hidden_size', '8', '--n-bases', '2',
            '--n-layers', '1', '--num_blocks', '1', '--diffusion_steps', '4',
            '--noise_schedule', 'cosine', '--add-static-graph', 'false']))
        args.num_nodes, args.num_rels = 6, 2
        config = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
            settings=settings(args), source_sha256=fingerprint([p for folder in ('src', 'diffusion', 'rgcn')
                for p in (root / folder).glob('*.py')], root),
            data_sha256=fingerprint([dataset / f'{x}.txt' for x in ('train', 'valid', 'test', 'stat')], dataset),
            history_sha256=fingerprint(list(cache.glob('snap_*.pkl')), dataset))
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'config.json').write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):
            model, gen = make_models(args, 6, 2, 0, 0, 3)
        torch.save(model.state_dict(), checkpoint / 'model.pt')
        torch.save(gen.state_dict(), checkpoint / 'diffusion.pt')
        torch.save(dict(version=2, provenance=config, completed_stage=2,
            model=model.state_dict(), diffusion=gen.state_dict()), checkpoint / 'latest.pt')
        before = {p: p.read_bytes() for folder in (dataset, checkpoint) for p in folder.rglob('*') if p.is_file()}
        options = parser().parse_args(['--gpu', '-1', '--run-dir', str(checkpoint),
            '--output', str(self.root / 'screen'), '--data-root', str(self.root), '--stages', '2'])
        with contextlib.redirect_stdout(io.StringIO()):
            run(options)
        summary = json.loads((options.output / 'summary.json').read_text())
        self.assertEqual(summary['overall']['original']['queries'], 2)
        self.assertEqual(set(summary['overall']), {'original', 'outgoing', 'incident', 'random'})
        for mode in ('outgoing', 'incident', 'random'):
            count = next(x for x in summary['counts'] if x['mode'] == mode)
            self.assertEqual(count['generated_facts'], [1, 1])
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
