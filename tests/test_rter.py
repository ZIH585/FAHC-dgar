"""Tiny CPU checks of causality, statistical controls and real main integration."""
import contextlib
import copy
import io
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from src import main, rter
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import rng_state, restore_rng
from test_official_protocol import run_loop

ROOT = Path(__file__).resolve().parents[1]


def history():
    return [np.array([[0,0,1],[2,1,3]],dtype=np.int64),
            np.array([[0,1,1],[2,0,3]],dtype=np.int64),
            np.array([[0,1,4],[2,0,5]],dtype=np.int64),
            np.array([[0,1,4]],dtype=np.int64)]


def rng_equal(a,b):
    return a['python']==b['python'] and np.array_equal(a['numpy'][1],b['numpy'][1]) and torch.equal(a['torch'],b['torch'])


class RTERTests(unittest.TestCase):
    def test_default_off_alias_and_invalid_args(self):
        p=build_arg_parser()
        self.assertEqual(validate_args(p.parse_args([])).rter_mode,'original')
        self.assertEqual(validate_args(p.parse_args(['--rter','--run-dir','/tmp/rter-toy'])).rter_mode,'transition')
        self.assertEqual(p.parse_args(['--rter','--rter-mode','original']).rter_mode,'original')
        for flags in (['--rter-beta','nan'],['--rter-horizon','0'],['--rter-window','0'],['--rter'],
                      ['--rter','--run-dir','/tmp/rter','--n-epochs','2']):
            with self.assertRaises(ValueError):validate_args(p.parse_args(flags))

    def test_prefix_never_reads_future_and_rejects_reverse_time(self):
        class Guard:
            def __len__(self):return 4
            def __getitem__(self,i):
                if i>=2:raise AssertionError('Future read')
                return history()[i]
        index=rter.RelationEvidence(Guard(),6,2,smoothing=1).advance(2)
        self.assertNotIn((1,4),index.events[0])
        self.assertEqual(index.stop,2)
        with self.assertRaises(ValueError):index.advance(1)
        with self.assertRaises(ValueError):index.advance(5)

    def test_known_relation_transition_has_lift_but_unseen_subject_does_not(self):
        index=rter.RelationEvidence(history(),6,2,smoothing=1).advance(2)
        evidence=index.evidence(np.array([[0,1],[5,1]]))
        scores=torch.zeros(2,6)
        modified=evidence.apply(scores,1)
        self.assertGreater(float(modified[0,1]),0)
        self.assertTrue(torch.equal(modified[1],scores[1]))
        self.assertTrue(torch.equal(scores,torch.zeros(2,6)))
        self.assertFalse(modified.requires_grad)

    def test_direction_augmentation_alone_produces_no_relation_lift(self):
        train=[np.array([[0,0,1]]),np.array([[0,0,1]])]
        index=rter.RelationEvidence(train,3,1,smoothing=1,channel='all').advance(2)
        np.testing.assert_array_equal(index.affinity(),np.zeros((2,2)))

    def test_window_and_time_ties_have_correct_transition_mass(self):
        train=[np.array([[0,0,1],[0,1,1]]),np.array([[0,1,1]]),np.empty((0,3),dtype=int),np.array([[0,0,1]])]
        index=rter.RelationEvidence(train,3,2,window=1).advance(4)
        self.assertEqual(index.transitions,2) # one pair and its inverse, not two same-time transitions
        self.assertAlmostEqual(index.counts.sum(),2*np.exp(-3/30))

    def test_incremental_rebuild_and_repeat_advance_match(self):
        a=rter.RelationEvidence(history(),6,2).advance(3)
        b=rter.RelationEvidence(history(),6,2).advance(1).advance(2).advance(2).advance(3)
        np.testing.assert_allclose(a.counts,b.counts,rtol=1e-15,atol=1e-15)
        np.testing.assert_allclose(a.popularity,b.popularity,rtol=1e-15,atol=1e-15)
        q=np.array([[0,1],[3,2]])
        torch.testing.assert_close(a.evidence(q).apply(torch.zeros(2,6),1),b.evidence(q).apply(torch.zeros(2,6),1))

    def test_empty_first_stage_singleton_dtype_and_zero_beta(self):
        index=rter.RelationEvidence(history(),6,2).advance(0)
        q=np.array([[0,1]]);s=torch.ones(1,6,dtype=torch.float64)
        self.assertIs(index.evidence(q).apply(s,1),s)
        self.assertEqual(index.evidence(np.empty((0,2),dtype=int)).shape,(0,6))
        index.advance(2)
        e=index.evidence(q)
        self.assertIs(e.apply(s,0),s)
        self.assertEqual(e.apply(s,.1).dtype,torch.float64)
        self.assertIs(index.evidence(q,'original').apply(s,1),s)
        with self.assertRaises(ValueError):e.apply(torch.ones(2,6),1)
        with self.assertRaises(ValueError):e.apply(torch.ones(1,6,dtype=torch.long),1)
        with self.assertRaises(ValueError):e.apply(s.float(),1e308)

    def test_encoding_invalid_input_and_no_answer_column(self):
        i=rter.RelationEvidence(history(),6,2).advance(2)
        for q in (np.array([[6,0]]),np.array([[0,4]]),np.array([[0.,1.]]),np.array([[0,1,3]])):
            with self.assertRaises(ValueError):i.evidence(q)
        with self.assertRaises(ValueError):rter.RelationEvidence([np.array([[0,4,1]])],6,2).advance(1)
        with self.assertRaises(ValueError):rter.RelationEvidence([np.array([[6,0,1]])],6,2).advance(1)

    def test_inverse_normalization_and_duplicate_facts(self):
        base=history()[:2]
        augmented=[np.concatenate((s,s,np.stack((s[:,2],s[:,1]+2,s[:,0]),axis=1))) for s in base]
        a=rter.RelationEvidence(base,6,2).advance(2)
        b=rter.RelationEvidence(augmented,6,2).advance(2)
        np.testing.assert_array_equal(a.counts,b.counts)
        self.assertEqual(a.events,b.events)
        np.testing.assert_array_equal(rter.known_queries(base[0],2),np.array([[0,0],[2,1],[1,2],[3,3]]))

    def test_controls_cross_channel_clip_and_no_rng_consumption(self):
        state=rng_state('cpu')
        index=rter.RelationEvidence(history(),6,2,smoothing=1,clip=.01).advance(3)
        q=np.array([[0,1],[3,2]])
        for mode in rter.MODES:
            evidence=index.evidence(q,mode)
            x=evidence.apply(torch.zeros(2,6),.3)
            self.assertTrue(torch.isfinite(x).all())
            self.assertLessEqual(float(x.max()),.00300001)
        self.assertTrue(rng_equal(state,rng_state('cpu')))
        np.testing.assert_array_equal(np.diag(index.affinity()),np.zeros(4))
        replica=rter.RelationEvidence(history(),6,2,smoothing=1,clip=.01).advance(3)
        np.testing.assert_array_equal(index.affinity(True),replica.affinity(True))

    def test_enabled_training_loop_is_identical_to_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,ar=run_loop(ROOT/'src/main.py',Path(tmp)/'a')
            b,br=run_loop(ROOT/'src/main.py',Path(tmp)/'b',rter_mode='transition')
            self.assertEqual(a,b)
            self.assertEqual(ar,br)

    def test_real_main_final_scores_logs_no_rng_or_model_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'SMALL/history_snap_v2';cache.mkdir(parents=True)
            train=history()
            for t,snap in enumerate(train):
                subjects=set(snap[:,0]);(cache/f'snap_{t}.pkl').write_bytes(pickle.dumps({str(s):set(map(tuple,snap[snap[:,0]==s])) for s in subjects}))
            args=validate_args(build_arg_parser().parse_args(['--gpu','-1','--dataset','SMALL','--data-root',str(root),
                '--run-dir',str(root),'--n-hidden','8','--hidden_size','8','--n-bases','2','--n-layers','1','--add-static-graph','false','--delete_feature_reply','true']))
            with contextlib.redirect_stdout(io.StringIO()):
                model=RecurrentRGCN('convtranse','uvrgcn',6,2,0,0,8,2,'sub',sequence_len=3,num_bases=2,num_hidden_layers=1,
                    self_loop=True,layer_norm=True,entity_prediction=True,relation_prediction=True,args=args).eval()
            valid=[np.array([[0,1,1]],dtype=int) for _ in train]
            ans=main.utils.load_all_answers_for_time_filter(valid,2,6,False)
            ansr=main.utils.load_all_answers_for_time_filter(valid,2,6,True)
            weights=copy.deepcopy(model.state_dict());state=rng_state('cpu');scores=[];rank=main.utils.get_total_rank
            def capture(triples,values,*a,**kw):
                if kw.get('rel_predict',0)==0:scores.append(values.clone())
                return rank(triples,values,*a,**kw)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(main.utils,'get_total_rank',capture):
                off=main.test(args,model,train,valid,2,6,False,ans,ansr,'unused',None,'valid',[],[],None,None)
            off_rng=rng_state('cpu');original=[s.clone() for s in scores];scores.clear();restore_rng(state,'cpu')
            args.rter_mode='transition';args.rter_smoothing=1;args.rter_beta=1
            with contextlib.redirect_stdout(io.StringIO()),patch.object(main.utils,'get_total_rank',capture):
                enabled=main.test(args,model,train,valid,2,6,False,ans,ansr,'unused',None,'valid',[],[],None,None,evidence_phase='VALIDATION')
            self.assertTrue(torch.equal(scores[0],original[0])) # no transition at t=1
            self.assertGreater(float((scores[1]-original[1]).abs().sum()),0)
            self.assertEqual(float(off[2]),float(enabled[2])) # relation ranking unaffected
            self.assertTrue(rng_equal(off_rng,rng_state('cpu')))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))
            logs=[json.loads(x) for x in (root/'rter.jsonl').read_text().splitlines()]
            self.assertEqual([x['history_stop'] for x in logs],[1,2,3])
            self.assertTrue(all(x['phase']=='VALIDATION' for x in logs))
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
