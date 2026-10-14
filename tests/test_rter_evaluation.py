"""End-to-end tiny saved-checkpoint smoke, using the real evaluation script."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from src.cli import build_arg_parser,validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import prepare_run,rng_state

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_rter as evaluation


class EvaluationTests(unittest.TestCase):
    def test_selection_never_tunes_on_confirmation_and_test_is_locked(self):
        rows=[]
        for stage in (1,2,3,4):
            row=dict(stage=stage,direction='forward',seen_before=False,original_rank=3)
            # Development prefers .1, confirmation would prefer 1. Do not let
            # confirmation labels choose the published beta.
            row['transition_b0.1_rank']=1 if stage<=2 else 3
            row['transition_b1_rank']=2 if stage<=2 else 1
            rows.append(row)
        summary=evaluation.summarize(rows,['transition'],[.1,1],5,2,'validation')
        self.assertEqual(summary['selection']['selected']['transition']['beta'],.1)
        self.assertFalse(summary['selection']['provisional_gate'])
        args=evaluation.parser().parse_args(['--run-dir','/tmp/unused','--output','/tmp/unused-other',
            '--split','test','--betas','.1,1'])
        with patch.object(evaluation,'load_original',side_effect=AssertionError('Must fail before load')):
            with self.assertRaises(ValueError):evaluation.run(args)

    def test_full_toy_script_matches_original_trace_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data/ICEWS14s';data.mkdir(parents=True)
            (data/'stat.txt').write_text('8\t2\t0\n')
            (data/'entity2id.txt').write_text(''.join(f'e{i}\t{i}\n' for i in range(8)))
            (data/'relation2id.txt').write_text('r0\t0\nr1\t1\n')
            snapshots=[]
            for t in range(8):
                snapshots.append(np.array([(i%6,(i+t)%2,(i+1+t)%8) for i in range(10)],dtype=np.int64))
            for name,times in (('train',range(6)),('valid',[6]),('test',[7])):
                (data/f'{name}.txt').write_text(''.join(f'{s}\t{r}\t{o}\t{t}\n' for t in times for s,r,o in snapshots[t]))
            cache=data/'history_snap_v2';cache.mkdir()
            for t,snap in enumerate(snapshots):
                train=snap[:8];(cache/f'snap_{t}.pkl').write_bytes(pickle.dumps(
                    {str(s):set(map(tuple,train[train[:,0]==s])) for s in set(train[:,0])}))
            run=root/'input';args=validate_args(build_arg_parser().parse_args([
                '--gpu','-1','--data-root',str(root/'data'),'--run-dir',str(run),
                '--n-hidden','8','--hidden_size','8','--n-bases','2','--n-layers','1',
                '--num_blocks','1','--add-static-graph','false','--diffusion_steps','4','--noise_schedule','cosine']))
            args.num_nodes,args.num_rels=8,2
            with contextlib.redirect_stdout(io.StringIO()):
                model=RecurrentRGCN('convtranse','uvrgcn',8,2,0,0,8,2,'sub',sequence_len=3,
                    num_bases=2,num_basis=args.n_basis,num_hidden_layers=1,dropout=args.dropout,
                    self_loop=True,layer_norm=True,input_dropout=args.input_dropout,
                    hidden_dropout=args.hidden_dropout,feat_dropout=args.feat_dropout,
                    weight=args.weight,discount=args.discount,angle=args.angle,
                    entity_prediction=True,relation_prediction=True,args=args)
                generator=Att_Diffuse_model(create_model_diffu(args),args,'uvrgcn',8,2,
                    dropout=args.dropout,max_time=8,num_bases=2)
            prepare_run(args)
            torch.save(model.state_dict(),run/'model.pt');torch.save(generator.state_dict(),run/'diffusion.pt')
            torch.save(dict(version=2,completed_stage=7,provenance=args._provenance,
                model=model.state_dict(),diffusion=generator.state_dict(),rng=rng_state('cpu')),run/'latest.pt')
            inputs={p:p.read_bytes() for p in [*run.iterdir(),*data.glob('*.txt'),*cache.iterdir()]}
            reports=[]
            for modes in ('repeat','transition,repeat,popularity,shuffled'):
                output=root/f'output-{len(reports)}'
                options=evaluation.parser().parse_args(['--gpu','-1','--data-root',str(root/'data'),
                    '--run-dir',str(run),'--output',str(output),'--betas','.1','--modes',modes])
                with contextlib.redirect_stdout(io.StringIO()),patch.object(evaluation.main,'tqdm',lambda x:x):
                    evaluation.run(options)
                summary=json.loads((output/'summary.json').read_text());reports.append(summary)
                self.assertTrue((output/'queries.csv').is_file())
                self.assertTrue((output/'rter.jsonl').is_file())
                self.assertEqual(summary['overall']['original_rank']['queries'],14)
            self.assertEqual(reports[0]['overall']['original_rank'],reports[1]['overall']['original_rank'])
            self.assertEqual(reports[0]['overall']['repeat_b0.1_rank'],reports[1]['overall']['repeat_b0.1_rank'])
            # Exercise the successful lock path with a synthetic fixture. This
            # forced gate has no empirical meaning and is never an experiment.
            locked=reports[1];locked['selection']['provisional_gate']=True
            selection_path=root/'locked.json';selection_path.write_text(json.dumps(locked))
            output=root/'locked-test'
            options=evaluation.parser().parse_args(['--gpu','-1','--data-root',str(root/'data'),
                '--run-dir',str(run),'--output',str(output),'--split','test','--selection-file',str(selection_path)])
            with contextlib.redirect_stdout(io.StringIO()),patch.object(evaluation.main,'tqdm',lambda x:x):
                evaluation.run(options)
            locked_result=json.loads((output/'summary.json').read_text())
            self.assertFalse(locked_result['selection']['performed'])
            self.assertEqual(locked_result['options']['betas'],'0.1')
            self.assertEqual(locked_result['options']['modes'],'transition')
            for p,original in inputs.items():self.assertEqual(p.read_bytes(),original)

    def test_failed_validation_selection_cannot_unlock_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json'
            path.write_text(json.dumps(dict(protocol='paired-rter-score-v1',options=dict(split='validation'),
                selection=dict(performed=True,provisional_gate=False))))
            args=evaluation.parser().parse_args(['--run-dir','/tmp/unused','--output','/tmp/other',
                '--split','test','--selection-file',str(path)])
            with patch.object(evaluation,'load_original',side_effect=AssertionError('Must fail before loading')):
                with self.assertRaises(ValueError):evaluation.run(args)


if __name__=='__main__':unittest.main()
