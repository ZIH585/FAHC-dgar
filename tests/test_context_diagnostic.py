"""CPU-only tests of label-free pooling intervention and real model connection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.context_diagnostic import (pool_for_query, selector_override, entity_scores,
                                        validate_config, ACCEPTED_ORIGINAL_SOURCES,
                                        parser, run, make_models)
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import PROTOCOL, UPSTREAM_COMMIT, settings, fingerprint
from rgcn.utils import build_sub_graph


class ContextDiagnosticTests(unittest.TestCase):
    def test_head_tail_inverse_and_count_control(self):
        # Entity 0 occurs as head/r0, head/r1, and tail/inverse(r0).
        tags = torch.tensor([[0, 0, 1], [0, 1, 2], [3, 0, 0], [4, 2, 0]])
        features = torch.arange(4*3*2, dtype=torch.float32).reshape(4, 3, 2)
        total, matched, vec, control = pool_for_query(tags, features, 0, 0, 2, 7)
        self.assertEqual((total, matched), (4, 2))  # tail of inverse r2 is role r0
        torch.testing.assert_close(vec, (features[0, 0]+features[3, 2])/2, rtol=0, atol=0)
        self.assertEqual(control.shape, vec.shape)
        a = pool_for_query(tags, features, 0, 2, 2, 7)
        self.assertEqual(a[:2], (4, 1))
        torch.testing.assert_close(a[2], features[2, 2], rtol=0, atol=0)
        self.assertEqual(pool_for_query(tags, features, 0, 3, 2, 7), (4, 0, None, None))
        self.assertEqual(pool_for_query(tags, features, 5, 0, 2, 7), (0, 0, None, None))
        again = pool_for_query(tags, features, 0, 0, 2, 7)
        torch.testing.assert_close(control, again[3], rtol=0, atol=0)

    def test_all_matching_and_empty_inputs(self):
        tags = torch.tensor([[0, 0, 1], [0, 0, 2]])
        features = torch.randn(2, 3, 8)
        total, matched, a, b = pool_for_query(tags, features, 0, 0, 2, 8)
        self.assertEqual((total, matched), (2, 2))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        empty = pool_for_query(tags[:0], features[:0], 0, 0, 2, 8)
        self.assertEqual(empty, (0, 0, None, None))

    def test_checkpoint_guard_rejects_other_methods(self):
        config = {'protocol': PROTOCOL, 'upstream_commit': UPSTREAM_COMMIT,
            'source_sha256': next(iter(ACCEPTED_ORIGINAL_SOURCES)),
            'settings': {'grb_mode': 'original', 'diffuc': 'add', 'delete_feature_reply': False}}
        validate_config(config)
        for key, value in [('grb_mode', 'adaptive'), ('grb_mode', 'fixed'), ('delete_feature_reply', True)]:
            bad = copy.deepcopy(config); bad['settings'][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad)
        bad = copy.deepcopy(config); bad['source_sha256'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_real_model_same_query_batch_and_no_parameters_or_rng_changes(self):
        args = validate_args(build_arg_parser().parse_args([
            '--gpu', '-1', '--n-hidden', '8', '--hidden_size', '8', '--n-bases', '2',
            '--n-layers', '1', '--add-static-graph', 'false']))
        with contextlib.redirect_stdout(io.StringIO()):
            model = RecurrentRGCN('convtranse', 'uvrgcn', 6, 2, 0, 0, 8, 2, 'sub',
                sequence_len=3, num_bases=2, num_hidden_layers=1,
                dropout=.2, self_loop=True, layer_norm=True,
                input_dropout=.2, hidden_dropout=.2, feat_dropout=.2,
                entity_prediction=True, relation_prediction=True, args=args).eval().requires_grad_(False)
        tags = torch.tensor([[0, 0, 1], [0, 1, 2], [3, 0, 0]])
        features = torch.randn(3, 3, 8)
        query = torch.tensor([[0, 0, 1], [1, 2, 0]])
        history = [build_sub_graph(6, 2, np.array([[0, 0, 1], [0, 1, 2]]), False, -1)]
        state = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
        entities, bank = model.select_entity(tags, features, False)
        reference = entity_scores(model, history, None, query, features, tags, False)
        with selector_override(model, entities, bank):
            same = entity_scores(model, history, None, query, features, tags, False)
        torch.testing.assert_close(same, reference, rtol=0, atol=0)
        _, _, match, _ = pool_for_query(tags, features, 0, 0, 2, 1)
        with mock.patch.object(model.decoder_ob.bn2, 'forward', wraps=model.decoder_ob.bn2.forward) as bn:
            with selector_override(model, entities, bank, 0, match):
                variant = entity_scores(model, history, None, query, features, tags, False)
                modified_entities, modified = model.select_entity(tags, features, False)
                torch.testing.assert_close(modified[1:], bank[1:], rtol=0, atol=0)
                torch.testing.assert_close(modified[0], match, rtol=0, atol=0)
            self.assertEqual(bn.call_count, 1)  # no accidental singleton decoder path
        self.assertEqual(tuple(variant.shape), (2, 6))
        self.assertFalse(variant.requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_whole_runner_synthetic_cpu_inference_no_optimizer(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp); data_dir = root/'data'/'SMALL'; cache = data_dir/'history_snap_v2'
            cache.mkdir(parents=True); checkpoint = root/'original'; checkpoint.mkdir()
            args = validate_args(build_arg_parser().parse_args(['--gpu', '-1', '--dataset', 'SMALL',
                '--n-hidden', '8', '--hidden_size', '8', '--n-bases', '2', '--n-layers', '1',
                '--num_blocks', '1', '--diffusion_steps', '4', '--noise_schedule', 'cosine',
                '--add-static-graph', 'false', '--data-root', str(root/'data')]))
            args.num_nodes, args.num_rels = 6, 2
            for name in ('train','valid','test','stat'):
                (data_dir/f'{name}.txt').write_text('synthetic\n')
            triples = [(0,0,1),(0,1,2),(0,0,3)]
            for stage, triple in enumerate(triples):
                (cache/f'snap_{stage}.pkl').write_bytes(pickle.dumps({'0': {triple}}))
            config = {'protocol': PROTOCOL, 'upstream_commit': UPSTREAM_COMMIT,
                'source_sha256': '6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538',
                'data_sha256': fingerprint(list(data_dir.glob('*.txt')),data_dir),
                'history_sha256': fingerprint(list(cache.glob('*.pkl')),data_dir), 'settings': settings(args)}
            model, generator = make_models(args,6,2,0,0,3)
            torch.save(model.state_dict(), checkpoint/'model.pt')
            torch.save(generator.state_dict(), checkpoint/'diffusion.pt')
            torch.save({'version':2,'completed_stage':2,'provenance':config,
                        'model':model.state_dict(),'diffusion':generator.state_dict()}, checkpoint/'latest.pt')
            (checkpoint/'config.json').write_text(json.dumps(config))
            data = types.SimpleNamespace(num_nodes=6,num_rels=2,
                train=np.array([[*triples[0],0]]*10),valid=np.array([[*triples[1],1]]*10),
                test=np.array([[*triples[2],2]]*10))
            before = {p.name:p.read_bytes() for p in checkpoint.iterdir()}
            options = parser().parse_args(['--gpu','-1','--run-dir',str(checkpoint),
                '--data-root',str(root/'data'),'--output',str(root/'diagnostic'),
                '--stages','2','--facts-per-stage','1'])
            with mock.patch('scripts.context_diagnostic.utils.load_data',return_value=data), \
                 mock.patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')):
                run(options)
            summary = json.loads((root/'diagnostic'/'summary.json').read_text())
            self.assertEqual(summary['queries'],2)
            self.assertEqual(set(summary['overall']),{'original','matched','control','no_replay'})
            self.assertEqual(before,{p.name:p.read_bytes() for p in checkpoint.iterdir()})
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
