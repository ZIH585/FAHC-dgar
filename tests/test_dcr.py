"""DCR CPU checks against the pinned sampler and actual small DGAR modules."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from diffusion import dcr
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'tests/reference/official_difffu_21.py.txt'


def reference_sampler():
    data = REFERENCE.read_bytes()
    manifest = json.loads((REFERENCE.parent / 'manifest.json').read_text())
    expected = next(x['sha'] for x in manifest['files'] if x['path'] == 'diffusion/difffu_21.py')
    assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected
    ns = {'__name__': 'pinned_author_diffusion'}
    exec(compile(data, str(REFERENCE), 'exec'), ns)
    return ns['DiffuRec']


class DCRTests(unittest.TestCase):
    def args(self):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8',
            '--n-bases', '2', '--n-layers', '1', '--num_blocks', '1',
            '--add-static-graph', 'false', '--diffusion_steps', '4',
            '--noise_schedule', 'cosine']))
        args.num_nodes, args.num_rels = 6, 2
        return args

    def modules(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            model = RecurrentRGCN('convtranse', 'uvrgcn', 6, 2, 0, 0, 8, 2,
                'sub', sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=args)
            gen = Att_Diffuse_model(create_model_diffu(args), args,
                'uvrgcn', 6, 2, max_time=4, dropout=.2)
        return model, gen

    def graph(self):
        return [build_sub_graph(6, 2, np.array([[0, 0, 1], [2, 1, 3]]), False, -1)]

    def generate(self, gen, model, args, tags):
        with torch.no_grad():
            return gen(None, tags, args, False, False, ct=2,
                       model=model, history_glist=self.graph())[1]

    def test_original_matches_pinned_author_values_buffers_and_rng(self):
        args = self.args()
        model, gen = self.modules(args)
        model.requires_grad_(False).train()
        gen.requires_grad_(False).train()
        old_model, old_gen = copy.deepcopy(model), copy.deepcopy(gen)
        pinned = reference_sampler()(args)
        pinned.load_state_dict(gen.diffu.state_dict())
        old_gen.diffu = pinned.requires_grad_(False).train()
        tags = torch.tensor([[0, 0, 1], [2, 1, 3], [1, 2, 0]])
        state = rng_state('cpu')
        old = self.generate(old_gen, old_model, args, tags)
        expected_rng = rng_state('cpu')
        restore_rng(state, 'cpu')
        new = self.generate(gen, model, args, tags)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng['torch']))
        self.assertEqual(rng_state('cpu')['python'], expected_rng['python'])
        for a, b in ((model, old_model), (gen, old_gen)):
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_mean_equals_original_without_exact_terminal_noise(self):
        args = self.args()
        model, gen = self.modules(args)
        model.requires_grad_(False).train()
        gen.requires_grad_(False).train()
        other_model, other_gen = copy.deepcopy(model), copy.deepcopy(gen)
        tags = torch.tensor([[0, 0, 1], [2, 1, 3]])
        captured = {}
        original_mean = gen.diffu.p_mean_variance

        def capture(*a, **kw):
            result = original_mean(*a, **kw)
            captured['mean'] = result[0].detach().clone()
            captured['x0'] = result[2].detach().clone()
            return result

        gen.diffu.p_mean_variance = capture
        state = rng_state('cpu')
        original = self.generate(gen, model, args, tags)
        expected_rng = rng_state('cpu')
        restore_rng(state, 'cpu')
        args.dcr_mode = 'mean'
        mean = self.generate(other_gen, other_model, args, tags)
        torch.testing.assert_close(mean[:, -1], captured['mean'], rtol=0, atol=0)
        torch.testing.assert_close(mean[:, :2], captured['x0'][:, :2], rtol=0, atol=0)
        torch.testing.assert_close(mean[:, :2], original[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(mean[:, -1], original[:, -1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng['torch']))
        self.assertIsNone(other_model.h.grad_fn)

    def test_guidance_is_partial_gradient_at_x0_not_posterior(self):
        args = self.args()
        model, _ = self.modules(args)
        model.eval()  # parameters still require gradients; grad() must not fill them
        x0 = torch.randn(2, 3, 8)
        tags = torch.tensor([[0, 0, 1], [2, 1, 3]])
        graph = self.graph()
        tail = x0[:, -1].detach().clone().requires_grad_(True)
        features = torch.cat((x0[:, :2], tail.unsqueeze(1)), dim=1)
        states, _, rels, _, _ = model(graph, None, False,
            features, tail, tags, diffu_rep='add')
        scores = model.decoder_ob(F.normalize(states[-1]), rels, tags, mode='test')
        expected = torch.autograd.grad(F.log_softmax(scores, -1).gather(1, tags[:, 2:3]).sum(), tail)[0]
        actual, gradient = dcr.guided_tail(model, x0, x0[:, -1], tags,
            graph, None, False, args)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, x0[:, -1] + args.classifier_scale * expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(model.h.grad_fn)
        with self.assertRaises(ValueError):
            dcr.guided_tail(model, x0, tail, tags, [], None, False, args)

    def test_raw_x0_keeps_network_output_and_skips_guide(self):
        args = self.args()
        args.dcr_mode = 'x0'
        model, gen = self.modules(args)
        model.requires_grad_(False).train()
        gen.requires_grad_(False).train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        capture = []
        handle = gen.diffu.xstart_model.register_forward_hook(
            lambda m, a, out: capture.append(out.detach().clone()))
        tags = torch.tensor([[0, 0, 1]])
        out = self.generate(gen, model, args, tags)
        handle.remove()
        self.assertEqual(len(capture), 1)
        torch.testing.assert_close(out, capture[0], rtol=0, atol=0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertFalse(out.requires_grad)

    def test_generator_validation_retains_original_even_when_dcr_enabled(self):
        args = self.args()
        args.dcr_mode, args.vl = 'guided-x0', True
        model, gen = self.modules(args)
        model.requires_grad_(False).train()
        gen.requires_grad_(False).train()
        old_model, old_gen = copy.deepcopy(model), copy.deepcopy(gen)
        pinned = reference_sampler()(args)
        pinned.load_state_dict(gen.diffu.state_dict())
        old_gen.diffu = pinned.requires_grad_(False).train()
        tags = torch.tensor([[0, 0, 1], [2, 1, 3]])
        state = rng_state('cpu')
        old = self.generate(old_gen, old_model, args, tags)
        expected_rng = rng_state('cpu')
        restore_rng(state, 'cpu')
        new = self.generate(gen, model, args, tags)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng['torch']))
        self.assertEqual(new.shape, (2, 8))

    def test_guided_endpoint_real_predictor_and_generator_backward_singleton(self):
        args = self.args()
        args.dcr_mode = 'guided-x0'
        model, gen = self.modules(args)
        gen.requires_grad_(False).train()
        guide = copy.deepcopy(model).requires_grad_(False).train()
        tags = torch.tensor([[0, 0, 1]])
        replay = self.generate(gen, guide, args, tags)
        self.assertEqual(replay.shape, (1, 3, 8))
        self.assertTrue(torch.isfinite(replay).all())
        self.assertIsNone(replay.grad_fn)
        self.assertTrue(all(p.grad is None for p in guide.parameters()))
        model.train()
        e, r, st, _, _ = model.get_loss(self.graph(), tags, None, False,
            model_output=replay, model_output_x_t=replay[:, -1], tag=tags,
            diffuc='add', output_reply_triple=tags)
        loss = .7*e + .3*r + st
        loss.backward()
        self.assertTrue(torch.isfinite(model.dynamic_emb.grad).all())
        model.eval()
        with torch.no_grad():
            triples, scores, rscores = model.predict(self.graph(), 2, None,
                tags, replay, replay[:, -1], tags, False)
        self.assertEqual(triples.tolist(), [[0, 0, 1], [1, 2, 0]])
        self.assertEqual(scores.shape, (2, 6))
        self.assertEqual(rscores.shape, (2, 4))
        self.assertTrue(torch.isfinite(scores).all())
        # The generator's existing current CE remains differentiable with the
        # new historical context as its conditioning input, no optimizer step.
        gen.requires_grad_(True).train()
        current = torch.tensor([[0, 0, 1], [1, 2, 0]])
        result = gen(None, current, args, True, False, ct=2,
            model_output=replay, targets=tags)
        ce = gen.loss_diffu_ce(result[1], current, result[7], current, result[4], result[5])
        self.assertTrue(torch.isfinite(ce))
        ce.backward()
        self.assertTrue(torch.isfinite(gen.emb_ent.grad).all())

    def test_empty_replay_and_recorder_detach_and_flush(self):
        args = self.args()
        args.dcr_mode = 'guided-x0'
        model, gen = self.modules(args)
        empty = torch.empty(0, 3, 8)
        out = dcr.sample(gen.diffu, model, torch.empty(0, 3, dtype=torch.long),
            empty, empty[:, -1], None, None, None, [], None, None, args, False)
        self.assertEqual(out.shape, (0, 3, 8))
        with tempfile.TemporaryDirectory() as tmp:
            recorder = dcr.Recorder(Path(tmp) / 'dcr.jsonl')
            a = torch.arange(5, dtype=torch.float32, requires_grad=True)
            recorder.observe(2, 'guided-x0', True, 2, a, 1e-7)
            recorder.observe(2, 'guided-x0', True, 1, a+3, 1e-7)
            self.assertIsNone(recorder.total.grad_fn)
            recorder.flush()
            data = json.loads(recorder.path.read_text())
            self.assertEqual(data['calls'], 2)
            self.assertEqual(data['facts'], 3)
            self.assertEqual(data['norms']['x0_head_norm'], 1.)
            self.assertIsNone(recorder.total)

    def test_defaults_and_cli_guardrails(self):
        p = build_arg_parser()
        self.assertEqual(validate_args(p.parse_args([])).dcr_mode, 'original')
        with tempfile.TemporaryDirectory() as tmp:
            args = validate_args(p.parse_args(['--dcr', '--run-dir', tmp]))
            self.assertEqual(args.dcr_mode, 'guided-x0')
            self.assertEqual(args._protocol, dcr.PROTOCOL)
            args.dcr_mode = 'original'
            validate_args(args)
            self.assertFalse(hasattr(args, '_protocol'))
            for flags in (['--delete_score', 'true'], ['--delete_feature_reply', 'true'],
                          ['--diffuc', 'false'], ['--classifier_scale', 'nan'], ['--n-epochs', '2'], ['--vl', 'true']):
                with self.assertRaises(ValueError):
                    validate_args(p.parse_args(['--dcr', '--run-dir', tmp] + flags))

    def test_first_stage_and_empty_hcp_use_existing_original_guards(self):
        args = self.args()
        args.dcr_mode = 'guided-x0'
        output = torch.tensor([[5, 0, 4]])
        with tempfile.TemporaryDirectory() as tmp:
            args.data_root, args.dataset = Path(tmp), 'SMALL'
            cache = Path(tmp) / 'SMALL/history_snap_v2'
            cache.mkdir(parents=True)
            import pickle
            (cache / 'snap_0.pkl').write_bytes(pickle.dumps({'0': {(0, 0, 1)}}))
            for stage in (1, 2):
                _, features, tags = Att_Diffuse_model.gereration_feature(None,
                    args, stage, None, None, None, None, None, output.numpy(),
                    output, False, [output.numpy()] * 3)
                self.assertIsNone(features)
                self.assertIsNone(tags)

    def test_actual_training_caller_supplies_guided_endpoint_to_first_loss(self):
        # Actual chronological loop, interrupted before its first backward or
        # optimizer step; a synthetic already-restored stage supplies history.
        from test_official_protocol import code_from
        from rgcn.utils import EarlyStopping
        import pickle
        args = self.args()
        args.dcr_mode = 'guided-x0'
        model, gen = self.modules(args)
        guide = copy.deepcopy(model).requires_grad_(False).train()
        gen.requires_grad_(False).train()
        train = [np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64)] * 4
        captured = []

        class ReachedLoss(Exception):
            pass

        def loss(*a, **kw):
            captured.append(kw)
            raise ReachedLoss()

        with tempfile.TemporaryDirectory() as tmp:
            args.data_root, args.dataset = Path(tmp), 'SMALL'
            args.run_dir, args.resume = Path(tmp), Path(tmp) / 'already-restored.pt'
            args._dcr_recorder = dcr.Recorder(Path(tmp) / 'dcr.jsonl')
            cache = Path(tmp) / 'SMALL/history_snap_v2'
            cache.mkdir(parents=True)
            (cache / 'snap_0.pkl').write_bytes(pickle.dumps({'0': {(0, 0, 1)}, '2': {(2, 1, 3)}}))
            torch.save(model.state_dict(), Path(tmp) / 'model.pt')
            ns = dict(args=args, model=model, model_repre=guide, model_diffpre=gen,
                first_stage=2, train_list=train, use_cuda=False, run_device=torch.device('cpu'),
                num_nodes=6, num_rels=2, static_graph=None, static_graph_diffu=None,
                model_state_file=Path(tmp) / 'model.pt', diffu_model_state_file=Path(tmp) / 'diffusion.pt',
                torch=torch, np=np, tqdm=lambda x:x, EarlyStopping=EarlyStopping,
                build_sub_graph=build_sub_graph)
            with mock.patch.object(model, 'get_loss', side_effect=loss):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ReachedLoss):
                        exec(code_from(ROOT / 'src/main.py', 'run_experiment', loop_only=True), ns)
            args._dcr_recorder.flush()
            record = json.loads((Path(tmp) / 'dcr.jsonl').read_text())
            self.assertEqual(record['mode'], 'guided-x0')
            self.assertEqual(record['stage'], 2)
            self.assertEqual(record['calls'], 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]['model_output'].shape, (2, 3, 8))
        self.assertFalse(captured[0]['model_output'].requires_grad)

    def test_diagnostic_arms_share_exact_x0_and_preserve_original_state(self):
        from scripts.diagnose_dcr import paired_endpoints
        args = self.args()
        model, gen = self.modules(args)
        model.requires_grad_(False).eval()
        gen.requires_grad_(False).train()
        tags = torch.tensor([[0, 0, 1], [2, 1, 3]])
        state = rng_state('cpu')
        expected = self.generate(gen, model, args, tags)
        expected_rng = rng_state('cpu')
        restore_rng(state, 'cpu')
        variants, stats = paired_endpoints(gen, model, args, tags, self.graph(), None, 2, False)
        # Generator dropout has no running state without a static graph, and
        # the predictor is eval. The capture hooks must not change Original.
        torch.testing.assert_close(variants['original'], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng['torch']))
        self.assertEqual(stats['denoiser_calls'], 1)
        for mode in ('original', 'mean', 'guided-x0'):
            torch.testing.assert_close(variants[mode][:, :2], variants['x0'][:, :2], rtol=0, atol=0)
            self.assertIsNone(variants[mode].grad_fn)
        self.assertNotIn('p_mean_variance', gen.diffu.__dict__)

    def test_default_linear_terminal_timestep_is_finite_for_each_endpoint(self):
        args = self.args()
        args.noise_schedule, args.diffusion_steps = 'linear', 35
        with contextlib.redirect_stderr(io.StringIO()):
            model, gen = self.modules(args)
        model.requires_grad_(False).eval()
        gen.requires_grad_(False).train()
        self.assertEqual(float(gen.diffu.betas[-1]), 1.)
        self.assertLess(float(gen.diffu.posterior_mean_coef1[-1]), 2e-7)
        tags = torch.tensor([[0, 0, 1], [2, 1, 3]])
        for mode in ('mean', 'x0', 'guided-x0'):
            args.dcr_mode = mode
            features = self.generate(gen, model, args, tags)
            self.assertEqual(features.shape, (2, 3, 8))
            self.assertTrue(torch.isfinite(features).all())

    def test_diagnostic_toy_checkpoint_end_to_end_read_only(self):
        from scripts.diagnose_dcr import make_models, parser, run
        from src.runtime import PROTOCOL, UPSTREAM_COMMIT, fingerprint, settings
        import pickle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'ICEWS14s'
            cache = data / 'history_snap_v2'
            cache.mkdir(parents=True)
            (data / 'entity2id.txt').write_text(''.join(f'e{i}\t{i}\n' for i in range(6)))
            (data / 'relation2id.txt').write_text('r0\t0\nr1\t1\n')
            (data / 'stat.txt').write_text('6\t2\n')
            pattern = [(0, 0, 1), (2, 1, 0), (1, 0, 3), (3, 1, 2)]
            for t, name in enumerate(('train', 'valid', 'test')):
                triples = [pattern[i%4] for i in range(10)]
                (data / f'{name}.txt').write_text(''.join(f'{s}\t{r}\t{o}\t{t+1}\n' for s,r,o in triples))
                d = {}
                for s,r,o in triples[:8]:
                    d.setdefault(str(s),set()).add((s,r,o))
                (cache / f'snap_{t}.pkl').write_bytes(pickle.dumps(d))
            args = self.args()
            args.data_root, args.dataset = root, 'ICEWS14s'
            with contextlib.redirect_stdout(io.StringIO()):
                model, gen = make_models(args, 6, 2, 0, 0, 3)
            config = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
                settings=settings(args), source_sha256=fingerprint([p for folder in ('src','diffusion','rgcn')
                    for p in (ROOT/folder).glob('*.py')], ROOT),
                data_sha256=fingerprint([data/f'{x}.txt' for x in ('train','valid','test','stat')], data),
                history_sha256=fingerprint(list(cache.glob('snap_*.pkl')), data))
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'config.json').write_text(json.dumps(config))
            torch.save(model.state_dict(), checkpoint / 'model.pt')
            torch.save(gen.state_dict(), checkpoint / 'diffusion.pt')
            torch.save(dict(version=2, provenance=config, completed_stage=2,
                model=model.state_dict(), diffusion=gen.state_dict()), checkpoint / 'latest.pt')
            before = {p:p.read_bytes() for folder in (data,checkpoint) for p in folder.rglob('*') if p.is_file()}
            options = parser().parse_args(['--gpu','-1','--run-dir',str(checkpoint),
                '--output',str(root/'screen'),'--data-root',str(root),'--stages','2'])
            with contextlib.redirect_stdout(io.StringIO()):
                run(options)
            result = json.loads((options.output/'summary.json').read_text())
            self.assertEqual(result['overall']['original']['queries'], 2)
            self.assertEqual(set(result['overall']), set(dcr.MODES))
            self.assertEqual(result['generation'][0]['denoiser_calls'], 1)
            self.assertEqual(result['generation'][0]['facts'], 1)
            self.assertTrue((options.output/'queries.csv').is_file())
            for p, contents in before.items():
                self.assertEqual(p.read_bytes(), contents)
            # New diagnostic runs on both configurations with the same tiny
            # synthetic weights. This checks loading/wiring, not performance.
            from scripts.diagnose_dcr_scale import parser as scale_parser, run as scale_run
            import copy
            for mode in ('original', 'mean'):
                input_run = checkpoint
                if mode == 'mean':
                    input_run = root / 'mean-checkpoint'
                    input_run.mkdir()
                    mean_config = copy.deepcopy(config)
                    mean_config['protocol'] = dcr.PROTOCOL
                    mean_config['settings']['dcr_mode'] = 'mean'
                    (input_run / 'config.json').write_text(json.dumps(mean_config))
                    torch.save(model.state_dict(), input_run / 'model.pt')
                    torch.save(gen.state_dict(), input_run / 'diffusion.pt')
                    torch.save(dict(version=2,provenance=mean_config,completed_stage=2,
                        model=model.state_dict(),diffusion=gen.state_dict()),input_run/'latest.pt')
                initial={p:p.read_bytes() for p in input_run.iterdir()}
                options=scale_parser().parse_args(['--gpu','-1','--checkpoint-mode',mode,
                    '--run-dir',str(input_run),'--output',str(root/f'scale-{mode}'),
                    '--data-root',str(root),'--stages','2'])
                with contextlib.redirect_stdout(io.StringIO()):
                    scale_run(options)
                result=json.loads((options.output/'summary.json').read_text())
                self.assertEqual(result['generation'][0]['denoiser_calls'],1)
                self.assertEqual(len(result['overall']),4)
                self.assertEqual(result['overall']['original']['queries'],2)
                for p,contents in initial.items():self.assertEqual(p.read_bytes(),contents)

    def test_scale_arms_match_per_fact_norms_and_keep_head_tokens(self):
        from scripts.diagnose_dcr_scale import scale_arms
        original=torch.randn(3,3,8,requires_grad=True)
        mean=torch.randn(3,8,requires_grad=True)*.01
        before=original.detach().clone()
        variants,stats=scale_arms(original,mean)
        torch.testing.assert_close(variants['scaled-original'][:,-1].norm(dim=-1),mean.norm(dim=-1))
        torch.testing.assert_close(variants['scaled-mean'][:,-1].norm(dim=-1),original[:,-1].norm(dim=-1))
        for value in variants.values():
            torch.testing.assert_close(value[:,:2],original[:,:2],rtol=0,atol=0)
            self.assertIsNone(value.grad_fn)
        torch.testing.assert_close(original.detach(),before,rtol=0,atol=0)
        self.assertEqual(stats['zero_mean_tails'],0)
        _,empty=scale_arms(torch.empty(0,3,8),torch.empty(0,8))
        self.assertEqual(empty['facts'],0)
        variants,zero=scale_arms(original[:1],torch.zeros(1,8))
        self.assertTrue(torch.isfinite(variants['scaled-mean']).all())
        self.assertEqual(zero['zero_mean_tails'],1)
        self.assertIsNone(zero['max_norm_mismatch']['scaled-mean'])

    def test_paired_scale_capture_preserves_sampler_output_rng_and_restores_hooks(self):
        from scripts.diagnose_dcr_scale import paired_tails
        args=self.args();model,gen=self.modules(args)
        model.requires_grad_(False).eval();gen.requires_grad_(False).train()
        tags=torch.tensor([[0,0,1],[2,1,3]])
        state=rng_state('cpu')
        expected=self.generate(gen,model,args,tags);expected_rng=rng_state('cpu')
        restore_rng(state,'cpu')
        args.dcr_mode='mean'
        calls=[]
        handle=gen.diffu.xstart_model.register_forward_hook(lambda *a:calls.append(1))
        try:variants,stats=paired_tails(gen,model,args,tags,self.graph(),None,2,False)
        finally:handle.remove()
        torch.testing.assert_close(variants['original'],expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng['torch']))
        self.assertEqual(len(calls),1)
        self.assertEqual(args.dcr_mode,'mean')
        self.assertNotIn('p_mean_variance',gen.diffu.__dict__)
        self.assertIsNone(model.h.grad_fn)


if __name__ == '__main__':
    unittest.main()
