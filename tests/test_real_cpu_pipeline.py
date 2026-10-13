"""Real DGAR modules on a handful of synthetic CPU triples (no training run)."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import types
import unittest

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.decoder import ConvTransR
from src.rrgcn import RecurrentRGCN
from src.runtime import restore_rng, rng_state

ROOT = Path(__file__).resolve().parents[1]


class RealCpuTests(unittest.TestCase):
    def test_frozen_no_grad_generators_match_upstream_train_mode(self):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8',
            '--n-bases', '2', '--n-layers', '1', '--num_blocks', '1',
            '--add-static-graph', 'false', '--diffusion_steps', '4',
            '--noise_schedule', 'cosine']))
        args.num_nodes, args.num_rels = 6, 2
        with contextlib.redirect_stdout(io.StringIO()):
            teacher = RecurrentRGCN('convtranse', 'uvrgcn', 6, 2, 0, 0, 8, 2,
                'sub', sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=args)
            generator = Att_Diffuse_model(create_model_diffu(args), args,
                                          'uvrgcn', 6, 2, max_time=4, dropout=.2)
        model = copy.deepcopy(teacher)
        frozen_teacher, frozen_generator = copy.deepcopy(teacher), copy.deepcopy(generator)
        frozen_teacher.requires_grad_(False).train()
        frozen_generator.requires_grad_(False).train()
        triples = np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64)
        history = [build_sub_graph(6, 2, triples, False, -1)]
        tags = torch.tensor([[0, 0, 1], [2, 1, 3], [1, 2, 0]], dtype=torch.long)
        state = rng_state('cpu')

        # This is the actual guided diffusion path. Its inner enable_grad must
        # keep gradients with respect to generated features despite no_grad.
        normal = generator(None, tags, args, False, False, ct=2,
                           model=teacher, history_glist=history)[1].detach()
        normal_rng = rng_state('cpu')
        restore_rng(state, 'cpu')
        with torch.no_grad():
            optimized = frozen_generator(None, tags, args, False, False, ct=2,
                model=frozen_teacher, history_glist=history)[1].detach()
        self.assertEqual(optimized.shape, (3, 3, 8))
        self.assertTrue(torch.isfinite(optimized).all())
        torch.testing.assert_close(optimized, normal, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), normal_rng['torch']))
        self.assertGreater(int(frozen_teacher.decoder_ob.bn0.num_batches_tracked), 0)
        for normal_model, frozen_model in ((teacher, frozen_teacher), (generator, frozen_generator)):
            for name, tensor in normal_model.state_dict().items():
                torch.testing.assert_close(tensor, frozen_model.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(optimized.requires_grad)
        self.assertTrue(all(p.grad is None for p in frozen_teacher.parameters()))

        # One backward through the real DGAR loss, with inverse query expansion.
        model.requires_grad_(True).train()
        loss_e, loss_r, loss_static, _, _ = model.get_loss(history,
            torch.from_numpy(triples[:1]), None, False, model_output=optimized,
            model_output_x_t=optimized[:, -1], tag=tags,
            diffuc='add', output_reply_triple=tags)
        loss = .7 * loss_e + .3 * loss_r + loss_static
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(torch.isfinite(model.dynamic_emb.grad).all())
        model.eval()
        with torch.no_grad():
            expanded, scores, rel_scores = model.predict(history, 2, None,
                torch.from_numpy(triples[:1]), optimized, optimized[:, -1], tags, False)
        self.assertEqual(tuple(expanded.shape), (2, 3))
        self.assertEqual(tuple(scores.shape), (2, 6))
        self.assertEqual(tuple(rel_scores.shape), (2, 4))
        self.assertTrue(torch.isfinite(scores).all())

    def test_hcp_selection_and_fact_order_match_upstream(self):
        source = ROOT / 'tests/reference/official_model_21.py.txt'
        manifest = json.loads((source.parent / 'manifest.json').read_text())
        expected = next(f['sha'] for f in manifest['files'] if f['path'] == 'diffusion/model_21.py')
        data = source.read_bytes()
        self.assertEqual(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(), expected)
        tree = ast.parse(data.decode())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Att_Diffuse_model')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'gereration_feature')
        ns = {'torch': torch, 'random': random, 'pickle': pickle}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), ns)
        original = ns['gereration_feature']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'data/SMALL/history_snap_v2'
            cache.mkdir(parents=True)
            train = [np.array([[0, 0, t % 6], [1, 1, (t + 1) % 6]]) for t in range(6)]
            for t, snap in enumerate(train):
                (cache / f'snap_{t}.pkl').write_bytes(pickle.dumps(
                    {str(s): {tuple(row) for row in snap if row[0] == s} for s in (0, 1)}))
            args = validate_args(build_arg_parser().parse_args([
                '--gpu', '-1', '--data-root', str(root / 'data'),
                '--dataset', 'SMALL', '--reply_batch', '2']))
            output = torch.tensor([[0, 0, 1], [1, 1, 2]])
            captured = []

            def generator(sequence, tag, *a, **kw):
                captured.append(tag.clone())
                return None, torch.rand(len(tag), 3, 2), None, None, None, None, None, None, None

            # Redirect only the upstream relative file root, leaving its actual
            # selector body untouched (default delete_his_prompt=False).
            def open_reference(path, mode):
                return (root / str(path).removeprefix('../')).open(mode)

            ns['open'] = open_reference
            state = rng_state('cpu')
            original(None, args, 5, generator, None, [], [], None,
                     output.numpy(), output, False, train)
            reference_rng = rng_state('cpu')
            restore_rng(state, 'cpu')
            Att_Diffuse_model.gereration_feature(None, args, 5, generator,
                None, [], [], None, output.numpy(), output, False, train)
            self.assertEqual(len(captured), 2)
            self.assertTrue(torch.equal(captured[0], captured[1]))
            self.assertEqual(random.getstate(), reference_rng['python'])
            self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng['torch']))

    def test_singleton_relation_eval_uses_upstream_batchnorm(self):
        decoder = ConvTransR(2, 8).eval()
        calls = []
        handle = decoder.bn2.register_forward_hook(lambda *a: calls.append(True))
        with torch.no_grad():
            scores = decoder(torch.randn(6, 8), torch.randn(4, 8), torch.tensor([[0, 0, 1]]))
        handle.remove()
        self.assertEqual(tuple(scores.shape), (1, 4))
        self.assertEqual(calls, [True])


if __name__ == '__main__':
    unittest.main()
