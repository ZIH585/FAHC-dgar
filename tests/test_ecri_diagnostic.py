"""State preservation and tiny CPU validation for the checkpoint diagnostic."""
import contextlib
import csv
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

from scripts import diagnose_ecri_gate as diagnostic
from src.runtime import fingerprint, settings
from test_ecri import args_for, make_model


class DiagnosticTests(unittest.TestCase):
    def test_interventions_restore_parameters_context_and_exceptions(self):
        model=make_model(args_for(True)).eval()
        features=torch.tensor([[.5,.3,.8]]*8)
        model.set_ecri_context(features,2)
        gate=model.ecri_gate; original=model._ecri_features
        alpha=model.alpha.detach().clone()
        with diagnostic.fusion_intervention(model,features,3,'alpha075'):
            self.assertIsNone(model.ecri_gate)
            self.assertEqual(float(model.alpha),.75)
        self.assertIs(model.ecri_gate,gate)
        self.assertIs(model._ecri_features,original)
        self.assertEqual(model._ecri_stage,2)
        torch.testing.assert_close(model.alpha,alpha,rtol=0,atol=0)
        with self.assertRaises(RuntimeError):
            with diagnostic.fusion_intervention(model,features,3,'current_only'):
                raise RuntimeError('synthetic interruption')
        self.assertIs(model.ecri_gate,gate)
        torch.testing.assert_close(model.alpha,alpha,rtol=0,atol=0)

    def test_shuffle_keeps_confidence_and_directed_pair_samples(self):
        features=np.array([[i/10,i/20,.6] for i in range(6)]+[[0,0,0]],dtype=np.float32)
        shuffled,moved=diagnostic.shuffled_features(features,20,7)
        np.testing.assert_array_equal(shuffled[:,2],features[:,2])
        np.testing.assert_array_equal(shuffled[-1],features[-1])
        self.assertEqual(sorted(map(tuple,shuffled[:6,:2])),sorted(map(tuple,features[:6,:2])))
        self.assertGreater(moved,0)
        again,_=diagnostic.shuffled_features(features,20,7)
        np.testing.assert_array_equal(shuffled,again)

    def test_one_synthetic_validation_slice_no_training_or_state_writes(self):
        args=args_for(True); args.num_nodes=8; args.num_rels=2
        data=types.SimpleNamespace(num_nodes=8,num_rels=2,
            train=np.array([[0,0,1,0]]*10),valid=np.array([[0,1,2,1]]*10),
            test=np.array([[0,0,3,2]]*10))
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            root=Path(temp); args.run_dir=root/'run';args.run_dir.mkdir()
            args.data_root=root/'data'; directory=args.data_root/args.dataset
            cache=directory/'history_snap_v2';cache.mkdir(parents=True)
            for name in ('train','valid','test','stat'):
                (directory/f'{name}.txt').write_text('synthetic fixture\n')
            for t,row in enumerate(((0,0,1),(0,1,2),(0,0,3))):
                (cache/f'snap_{t}.pkl').write_bytes(pickle.dumps({'0':{row}}))
            # Construct tiny real modules; save their initialized weights only.
            model=make_model(args)
            from diffusion.model_21 import Att_Diffuse_model,create_model_diffu
            diffusion=Att_Diffuse_model(create_model_diffu(args),args,args.encoder,8,2,
                max_time=3,num_words=0,num_static_rels=0,num_bases=2)
            torch.save(model.state_dict(),args.run_dir/'model.pt')
            torch.save(diffusion.state_dict(),args.run_dir/'diffusion.pt')
            config={'settings':settings(args),'source_sha256':diagnostic.source_manifest()['source_sha256'],
                'data_sha256':fingerprint([directory/f'{n}.txt' for n in ('train','valid','test','stat')],directory),
                'history_sha256':fingerprint(list(cache.glob('snap_*.pkl')),directory)}
            (args.run_dir/'config.json').write_text(json.dumps(config))
            before={p.name:p.read_bytes() for p in args.run_dir.iterdir()}
            options=diagnostic.build_parser().parse_args(['--run-dir',str(args.run_dir),
                '--data-root',str(args.data_root),'--output-dir',str(root/'output'),
                '--stages','2','--query-limit','1','--shuffle-repeats','2'])
            with mock.patch.object(diagnostic.utils,'load_data',return_value=data):
                diagnostic.run(options)
            summary=json.loads((root/'output/summary.json').read_text())
            self.assertEqual(summary['split'],'VALIDATION')
            self.assertEqual(summary['stage_records'][0]['hcp_facts'],2)
            self.assertEqual(summary['pooled_metrics']['full']['queries'],2)
            self.assertEqual(len(summary['pooled_metrics']),8)
            self.assertTrue(summary['all_parameter_grads_none'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in args.run_dir.iterdir()})
            with (root/'output/queries.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([r['inverse'] for r in rows],['0','1'])
            self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
