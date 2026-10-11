"""Synthetic CPU-only checks; no dataset loading or training epochs."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from src.cli import build_arg_parser, validate_args
from src.ecri import EvolutionConfig, EvolutionGate, EvolutionHistory, record_stage
from src.rrgcn import RecurrentRGCN
from src.runtime import (isolated_rng, save_training_state, restore_training_state,
                         rng_state, restore_rng, validate_history_cache)
from rgcn.utils import build_sub_graph, load_all_answers_for_time_filter, EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'f7bf2d8'
torch.set_num_threads(1)


def args_for(enabled=False):
    tokens = ['--gpu','-1','--n-hidden','8','--hidden_size','8','--n-bases','2',
              '--n-basis','2','--add-static-graph','false','--diffusion_steps','2',
              '--dropout','0','--input-dropout','0','--hidden-dropout','0','--feat-dropout','0',
              '--ecri-window','1','--ecri-min-support','1','--ecri-sample-cap','8',
              '--ecri-permutations','3','--run-dir','/tmp/ecri-unit-unused']
    if enabled:
        tokens.append('--ecri')
    return validate_args(build_arg_parser().parse_args(tokens))


def make_model(args, cls=RecurrentRGCN, static=False):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls('convtranse','uvrgcn',8,2,1 if static else 0,2 if static else 0,
                   8,2,'sub',3,num_bases=2,num_basis=2,num_hidden_layers=2,
                   self_loop=True,layer_norm=True,use_static=static,entity_prediction=True,
                   relation_prediction=True,use_cuda=False,gpu=-1,args=args)


def graph():
    return build_sub_graph(8,2,np.array([[0,0,1],[2,1,3],[0,1,4]],dtype=np.int64),False,-1)


def example_loss(model, replay=True, empty=False, static=None):
    triples=torch.tensor([[0,0,1]])
    rep=torch.arange(24,dtype=torch.float32).reshape(1,3,8)/24
    tag=triples if not empty else torch.empty((0,3),dtype=torch.long)
    if empty:
        rep=rep[:0]
    return model.get_loss([graph(),graph()],triples,static,False,
                         model_output=rep if replay else None,
                         model_output_x_t=rep[:,-1] if replay else None,
                         tag=tag,diffuc='add',output_reply_triple=tag)


class StatisticsTests(unittest.TestCase):
    def history(self, **kw):
        return EvolutionHistory(8,2,EvolutionConfig(window=1,min_support=2,sample_cap=8,
                                                    permutations=10,**kw))

    def test_warmup_empty_and_small_support(self):
        h=self.history()
        np.testing.assert_array_equal(h.features(0),np.zeros((8,3)))
        h.observe(0,[])
        np.testing.assert_array_equal(h.features(1),np.zeros((8,3)))
        h.observe(1,np.array([[0,0,1]],dtype=np.int64))
        np.testing.assert_array_equal(h.features(2),np.zeros((8,3)))
        self.assertEqual(len(h._snapshots),2)

    def test_complete_windows_required(self):
        h=EvolutionHistory(8,2,EvolutionConfig(window=2,min_support=1))
        x=np.array([[0,0,1]]*10)
        for t in range(3): h.observe(t,x)
        self.assertTrue(h.features(3).sum()==0)
        h.observe(3,x)
        self.assertGreater(h.features(4)[0,2],0)

    def test_detects_change_and_preserves_rng(self):
        h=self.history()
        h.observe(0,np.array([[0,0,1]]*8))
        h.observe(1,np.array([[0,1,2]]*8))
        random.seed(15); before=random.getstate()
        f=h.features(2)
        self.assertGreater(f[0,0],.5)
        self.assertGreater(f[0,1],.5)
        self.assertGreater(f[0,2],0)
        self.assertEqual(before,random.getstate())
        np.testing.assert_array_equal(f,h.features(2))

    def test_directions_and_sample_cap(self):
        h=self.history()
        h.observe(0,np.array([[0,0,1],[0,0,1],[0,0,1]]))
        h.observe(1,np.array([[0,1,1],[0,1,1],[0,1,1]]))
        f=h.features(2)
        # Subject and inverse-object roles both change. Entity-local permutation
        # draws need not produce identical null estimates for the two endpoints.
        for entity in (0,1):
            self.assertGreater(float(f[entity,0]),.5)
            self.assertGreater(float(f[entity,1]),.5)
            self.assertAlmostEqual(float(f[entity,2]),3/23,places=6)
        np.testing.assert_array_equal(f[2:],np.zeros((6,3)))

    def test_strict_causality_and_bounded_history(self):
        h=self.history()
        x=np.array([[0,0,1],[1,1,2]])
        with self.assertRaises(ValueError): h.observe(1,x)
        for t in range(6):
            self.assertEqual(h.next_stage,t)
            h.features(t)
            h.observe(t,x)
            self.assertLessEqual(len(h._snapshots),2)
        with self.assertRaises(ValueError): h.features(5)
        with self.assertRaises(ValueError): h.features(7)
        state=h.state_dict(); saved=h.features(6).copy()
        h.observe(6,np.array([[4,1,5]]*40))
        restored=self.history(); restored.load_state_dict(state)
        np.testing.assert_array_equal(saved,restored.features(6))

    def test_validation_and_atomic_restore(self):
        h=self.history()
        for bad in (np.array([[0,2,1]]),np.array([[8,0,1]]),np.array([[-1,0,1]]),
                    np.array([[0.,0.,1.]]),np.zeros((0,4),dtype=int)):
            with self.assertRaises(ValueError): h.observe(0,bad)
        self.assertEqual(h.next_stage,0)
        h.observe(0,np.array([[0,0,1]]))
        state=h.state_dict(); state['snapshots'][0]=(2,state['snapshots'][0][1])
        with self.assertRaises(ValueError): h.load_state_dict(state)
        self.assertEqual(h.next_stage,1)


class ModelTests(unittest.TestCase):
    def test_disabled_matches_clean_baseline_output_gradient_and_step(self):
        source=subprocess.check_output(['git','show',f'{BASELINE}:src/rrgcn.py'],cwd=ROOT,text=True)
        module=types.ModuleType('original_reference'); exec(compile(source,'baseline_rrgcn.py','exec'),module.__dict__)
        args=args_for(False)
        torch.default_generator.manual_seed(17); base=make_model(args,module.RecurrentRGCN)
        torch.default_generator.manual_seed(17); new=make_model(args)
        self.assertEqual(set(base.state_dict()),set(new.state_dict()))
        for key,value in base.state_dict().items(): torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        opt=[torch.optim.Adam(m.parameters(),lr=.001) for m in (base,new)]
        outputs=[]
        for model,optimizer in zip((base,new),opt):
            torch.default_generator.manual_seed(99)
            loss=example_loss(model); total=sum(loss[:3]); total.backward()
            outputs.append([v.detach().clone() for v in loss])
            optimizer.step()
        for a,b in zip(*outputs): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(base.parameters(),new.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_gate_fallback_monotonic_and_gradients(self):
        gate=EvolutionGate()
        alpha=torch.tensor(1.4,requires_grad=True)
        empty=torch.zeros((2,3))
        torch.testing.assert_close(gate(alpha,empty),torch.full((2,1),1.4),rtol=0,atol=0)
        features=torch.tensor([[0.,0.,1.],[.8,.7,1.]])
        self.assertGreater(gate(alpha,features)[1],gate(alpha,features)[0])
        gate(alpha,features).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters()))

    def test_enabled_real_model_loss_backward_and_no_eval_grad(self):
        model=make_model(args_for(True))
        features=torch.tensor([[.8,.6,.8]]*8,requires_grad=True)
        model.set_ecri_context(features,2)
        loss=example_loss(model); sum(loss[:3]).backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(model._ecri_features.grad_fn)
        for p in model.ecri_gate.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs()),0)
        model.zero_grad(set_to_none=True); model.eval()
        rep=torch.ones(1,3,8); triples=torch.tensor([[0,0,1]])
        _,scores,relations=model.predict([graph()],2,None,triples,rep,rep[:,-1],triples,False)
        self.assertEqual(tuple(scores.shape),(2,8)); self.assertEqual(tuple(relations.shape),(2,4))
        self.assertFalse(scores.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertNotIn('_ecri_features',model.state_dict())
        self.assertEqual(sum(k.startswith('ecri_gate.') for k in model.state_dict()),3)

    def test_empty_replay_and_missing_context(self):
        model=make_model(args_for(True))
        for losses in (example_loss(model,replay=False),example_loss(model,empty=True)):
            self.assertTrue(torch.isfinite(sum(losses[:3])).all())
        with self.assertRaises(RuntimeError): example_loss(model)
        with self.assertRaises(ValueError): model.set_ecri_context(torch.zeros(7,3),2)
        with self.assertRaises(ValueError): model.set_ecri_context(torch.full((8,3),float('nan')),2)

    def test_default_static_graph_path(self):
        model=make_model(args_for(True),static=True)
        model.set_ecri_context(torch.tensor([[.2,.4,.5]]*8),2)
        static=build_sub_graph(10,1,np.array([[0,0,8],[1,0,9],[2,0,8]]),False,-1)
        losses=example_loss(model,static=static)
        sum(losses[:3]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_real_diffusion_guidance_bypasses_gate(self):
        from diffusion.model_21 import create_model_diffu,Att_Diffuse_model
        args=args_for(True); args.num_blocks=1; args.num_nodes=8; args.num_rels=2
        model=make_model(args).eval()  # Deliberately no ECRI context for the guider.
        with contextlib.redirect_stdout(io.StringIO()):
            diffusion=Att_Diffuse_model(create_model_diffu(args),args,'uvrgcn',8,2,num_bases=2,
                                       max_time=4,num_words=0,num_static_rels=0).eval()
        triples=torch.tensor([[0,0,1]])
        with torch.no_grad():
            result=diffusion(None,triples,args,False,False,model=model,history_glist=[graph()])
        self.assertEqual(tuple(result[1].shape),(1,3,8))
        self.assertTrue(torch.isfinite(result[1]).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # A one-triple diffusion CE forward/backward, without an epoch or update.
        diffusion.train()
        result=diffusion(None,triples,args,True,False)
        loss=diffusion.loss_diffu_ce(result[1],triples,emb_ent=result[5])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in diffusion.parameters() if p.grad is not None))


class RuntimeTests(unittest.TestCase):
    def test_cache_must_match_training_snapshot(self):
        train=[np.array([[0,0,1]])]
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp); path=directory/'snap_0.pkl'
            with self.assertRaises(FileNotFoundError): validate_history_cache(train,2,directory)
            for cache in ({'0':{(0,0,1)}},{'0':{(0,0,1)},'1':{(1,2,0)}}):
                path.write_bytes(pickle.dumps(cache)); validate_history_cache(train,2,directory)
            for cache in ({},{'0':{(0,0,1),(0,1,2)}},{'1':{(0,0,1)}}):
                path.write_bytes(pickle.dumps(cache))
                with self.assertRaises(ValueError): validate_history_cache(train,2,directory)

    def test_configuration_contract(self):
        parser=build_arg_parser()
        self.assertFalse(parser.parse_args([]).ecri)
        self.assertFalse(any('fahc' in a.dest or a.dest=='hcp_selection' for a in parser._actions))
        for options in (['--ecri'],['--ecri','--run-dir','/tmp/x','--diffuc','false'],
                        ['--ecri','--run-dir','/tmp/x','--ecri-kappa','0'],
                        ['--ecri','--run-dir','/tmp/x','--n-epochs','2']):
            with self.assertRaises(ValueError): validate_args(parser.parse_args(options))
        for option in ('--fahc-fact-budget','--hcp-selection'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parser.parse_args([option,'1'])
        self.assertEqual(args_for(True).ecri_window,1)

    def test_rng_isolation(self):
        torch.default_generator.manual_seed(5); random.seed(5); np.random.seed(5)
        state=rng_state('cpu')
        with isolated_rng(82,'cpu'):
            torch.rand(4); random.random(); np.random.rand()
        draws=(torch.rand(4),random.random(),np.random.rand())
        restore_rng(state,'cpu')
        expected=(torch.rand(4),random.random(),np.random.rand())
        torch.testing.assert_close(draws[0],expected[0]); self.assertEqual(draws[1:],expected[1:])

    def test_best_adam_and_stage_resume(self):
        args=args_for(True)
        model=make_model(args); diffusion=torch.nn.Linear(2,2)
        teacher=make_model(args); diff_teacher=torch.nn.Linear(2,2)
        optimizer=torch.optim.Adam(model.parameters()); diff_optimizer=torch.optim.Adam(diffusion.parameters())
        h=EvolutionHistory(8,2,EvolutionConfig.from_args(args))
        h.observe(0,np.array([[0,0,1]])); h.observe(1,np.array([[0,1,2]]))
        model.set_ecri_context(h.features(2),2)
        sum(example_loss(model)[:3]).backward(); optimizer.step(); optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as temp:
            args.run_dir=Path(temp); args._provenance={'test':'synthetic'}
            stopper=EarlyStopping(path=str(args.run_dir/'model.pt'))
            stopper(.6,model,optimizer)
            initial=copy.deepcopy(model.state_dict())
            with torch.no_grad(): model.alpha.add_(2)
            stopper(.4,model,optimizer)
            model.load_state_dict(torch.load(args.run_dir/'model.pt',weights_only=True))
            optimizer.load_state_dict(stopper.optimizer_state)
            save_training_state(args,1,model,diffusion,optimizer,diff_optimizer,'cpu',h.state_dict())
            expected_draw=torch.rand(5)
            args.resume=args.run_dir/'latest.pt'
            start,aux=restore_training_state(args,model,diffusion,teacher,diff_teacher,optimizer,diff_optimizer,'cpu')
            self.assertEqual(start,2)
            torch.testing.assert_close(torch.rand(5),expected_draw,rtol=0,atol=0)
            restored=EvolutionHistory(8,2,EvolutionConfig.from_args(args)); restored.load_state_dict(aux)
            np.testing.assert_array_equal(restored.features(2),h.features(2))
            for key,value in initial.items(): torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
            self.assertGreater(len(optimizer.state),0)
            record_stage(args.run_dir/'ecri.jsonl',restored,model.ecri_gate)
            self.assertEqual(json.loads((args.run_dir/'ecri.jsonl').read_text())['prefix_last_snapshot'],1)

    def test_final_evaluation_rebuilds_prefix_and_uses_past_graph(self):
        import src.main as main
        args=args_for(True); args.reply_batch=10
        train=[np.array([[0,0,1]]*3),np.array([[0,1,2]]*3),np.array([[0,0,3]]*3),np.array([[0,1,4]]*3)]
        valid=[np.array([[0,0,1]]) for _ in train]
        model=make_model(args).eval(); captured=[]; guidance=[]
        original=model.set_ecri_context
        def capture(features,stage):
            captured.append((stage,features.copy())); original(features,stage)
        model.set_ecri_context=capture
        class Generator:
            def __call__(self,sequence,tag,*unused,**kwargs):
                guidance.append((kwargs['ct'],len(kwargs['history_glist'])))
                rep=torch.ones(len(tag),3,8)
                return None,rep,None,None,None,None,None,None,None
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            args.data_root=Path(temp); cache=args.data_root/args.dataset/'history_snap_v2'; cache.mkdir(parents=True)
            for t,rows in enumerate(train):
                with (cache/f'snap_{t}.pkl').open('wb') as stream: pickle.dump({'0':set(map(tuple,rows))},stream)
            answers=load_all_answers_for_time_filter(valid,2,8,False)
            answers_r=load_all_answers_for_time_filter(valid,2,8,True)
            # A deliberately invalid supplied last_history must never be used.
            main.test(args,model,train,valid,2,8,False,answers,answers_r,'unused',None,
                      'validation',['FUTURE_GRAPH'],None,None,Generator())
        reference=EvolutionHistory(8,2,EvolutionConfig.from_args(args)); reference.observe(0,train[0])
        self.assertEqual([t for t,_ in captured],[1,2,3])
        for t,features in captured:
            np.testing.assert_array_equal(features,reference.features(t)); reference.observe(t,train[t])
        self.assertEqual(guidance,[(1,1),(2,2),(3,3)])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_training_call_order_is_wired(self):
        tree=ast.parse((ROOT/'src/main.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_experiment')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
        def lines(name): return sorted(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr==name)
        self.assertLess(lines('set_ecri_context')[0],lines('get_loss')[0])
        self.assertLess(lines('get_loss')[0],lines('observe')[-1])
        saves=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='save_training_state']
        self.assertGreater(saves[0].lineno,lines('observe')[-1])

    def test_training_loop_stops_at_first_synthetic_batch(self):
        import src.main as main
        args=args_for(True)
        data=types.SimpleNamespace(num_nodes=8,num_rels=2,
            train=np.array([[0,0,1,0]]*10),valid=np.array([[0,1,2,1]]*10),
            test=np.array([[0,0,3,2]]*10))
        original=RecurrentRGCN.get_loss
        calls=[]
        class FirstBatchReached(Exception): pass
        def stop_after_loss(model,*positional,**keywords):
            self.assertEqual(model._ecri_stage,1)
            self.assertEqual(tuple(model._ecri_features.shape),(8,3))
            self.assertEqual(float(model._ecri_features.sum()),0.)
            losses=original(model,*positional,**keywords)
            self.assertTrue(torch.isfinite(sum(losses[:3])).all())
            calls.append(True)
            raise FirstBatchReached()
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            args.data_root=Path(temp)/'data'; args.run_dir=Path(temp)/'run'
            directory=args.data_root/args.dataset; cache=directory/'history_snap_v2'; cache.mkdir(parents=True)
            for name in ('train','valid','test','stat'): (directory/f'{name}.txt').write_text('synthetic fixture\n')
            for t,row in enumerate(((0,0,1),(0,1,2),(0,0,3))):
                (cache/f'snap_{t}.pkl').write_bytes(pickle.dumps({'0':{row}}))
            with mock.patch.object(main.utils,'load_data',return_value=data), \
                 mock.patch.object(main,'fitlog'), \
                 mock.patch.object(RecurrentRGCN,'get_loss',stop_after_loss), \
                 self.assertRaises(FirstBatchReached):
                main.run_experiment(args)
            self.assertFalse((args.run_dir/'latest.pt').exists())
            self.assertFalse((args.run_dir/'model.pt').exists())
        self.assertEqual(calls,[True])

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
