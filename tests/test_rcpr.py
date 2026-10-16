"""RCPR correctness, integration and OFF protocol tests on tiny CPU tensors."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn.utils import build_sub_graph
from src.cli import build_arg_parser, validate_args
from src.decoder import ConvTransE
from src.rcpr import SharedAdapter, ScoreResidual, make_context, make_layout, context_statistics
from src.rcpr_probe import fit_head, gate
from src.rcpr_diagnostics import EvaluationRecorder, record_training_stage
from src.rrgcn import RecurrentRGCN
from src.runtime import settings, rng_state, restore_rng, save_training_state, restore_training_state

ROOT=Path(__file__).resolve().parents[1]


def args(mode='original'):
    a=validate_args(build_arg_parser().parse_args(['--gpu','-1','--seed','1','--n-hidden','8',
        '--hidden_size','8','--n-bases','2','--n-layers','1','--num_blocks','1','--add-static-graph','false',
        '--diffusion_steps','4','--noise_schedule','cosine','--run-dir','/tmp/rcpr-test',
        '--rcpr-mode',mode,'--rcpr-rank','2','--rcpr-adapter-rank','2','--rcpr-score-chunk','2']))
    a.num_nodes,a.num_rels=6,2
    return a


def model(a,cls=RecurrentRGCN):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls('convtranse','uvrgcn',6,2,0,0,8,2,'sub',sequence_len=3,num_bases=2,
            num_hidden_layers=1,dropout=.2,self_loop=True,layer_norm=True,input_dropout=.2,
            hidden_dropout=.2,feat_dropout=.2,entity_prediction=True,relation_prediction=True,args=a)


def history():
    return [build_sub_graph(6,2,np.array([[0,0,1],[2,1,3],[1,1,0]]),False,-1)]


def tags():
    return torch.tensor([[0,0,1],[0,1,1],[1,2,0],[2,1,3],[0,0,0],[5,3,1]])


class RCPRTests(unittest.TestCase):
    def assert_tensors(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,msg=k)

    def test_off_exact_baseline_init_loss_gradients_scores_rng(self):
        ref=ROOT/'tests/reference/rcpr_baseline_rrgcn.py.txt'
        metadata=json.loads((ref.parent/'rcpr_baseline.json').read_text())
        self.assertEqual(hashlib.sha256(ref.read_bytes()).hexdigest(),metadata['files']['rrgcn'])
        ns={'__name__':'rcpr_reference'};exec(compile(ref.read_bytes(),str(ref),'exec'),ns)
        a=args();state=rng_state('cpu');expected=model(a,ns['RecurrentRGCN']);init=torch.get_rng_state()
        restore_rng(state,'cpu');actual=model(a)
        self.assertIsNone(actual.rcpr_head);self.assertIsNone(actual.rcpr_adapter)
        self.assert_tensors(actual.state_dict(),expected.state_dict());self.assertTrue(torch.equal(init,torch.get_rng_state()))
        features=torch.randn(len(tags()),3,8);state=rng_state('cpu');outputs=[]
        for m in [expected,actual]:
            restore_rng(state,'cpu')
            losses=m.get_loss(history(),tags()[:1],None,False,model_output=features,model_output_x_t=features[:,2],
                              tag=tags(),diffuc='add',output_reply_triple=tags())
            loss=.7*losses[0]+.3*losses[1]+losses[2];loss.backward();m.eval()
            pred=m.predict(history(),2,None,tags()[:1],features,features[:,2],tags(),False)
            outputs.append((loss.detach(),pred[1],pred[2],torch.get_rng_state()))
        for x,y in zip(*outputs):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assert_tensors(actual.state_dict(),expected.state_dict())
        for p,q in zip(actual.parameters(),expected.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        self.assertFalse(any(k.startswith('rcpr_') for k in settings(a)))

    def test_decoder_query_matches_original_and_ignores_target(self):
        ref=ROOT/'tests/reference/rcpr_baseline_decoder.py.txt'
        ns={};exec(compile(ref.read_bytes(),str(ref),'exec'),ns)
        d=ConvTransE(6,8,.2,.2,.2);original=ns['ConvTransE'](6,8,.2,.2,.2);original.load_state_dict(d.state_dict())
        e,r=torch.randn(6,8),torch.randn(4,8)
        for mode in ['train','eval']:
            getattr(d,mode)();getattr(original,mode)();state=rng_state('cpu')
            expected=original(e,r,tags());restore_rng(state,'cpu')
            scores,query=d(e,r,tags(),return_query=True)
            torch.testing.assert_close(scores,expected,rtol=0,atol=0)
            torch.testing.assert_close(scores,query@torch.tanh(e).T,rtol=0,atol=0)
        t=tags().clone();t[:,2]=(t[:,2]+1)%6
        d.eval();x=d(e,r,tags(),return_query=True);y=d(e,r,t,return_query=True)
        for left,right in zip(x,y):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_zero_initialization_live_gradients_and_rng(self):
        state=torch.get_rng_state().clone();adapter=SharedAdapter(8,2);head=ScoreResidual(8,2,2)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertGreater(float(adapter.down.weight.norm()),0);self.assertGreater(float(head.prototype.weight.norm()),0)
        z=torch.randn(2*len(tags()),8);layout=make_layout(tags(),6,2);c=make_context(layout,adapter(z))
        scores=torch.randn(len(tags()),6);q=torch.randn(len(tags()),8)
        actual=head(scores,q,tags()[:,:2],c);torch.testing.assert_close(actual,scores,rtol=0,atol=0)
        F.cross_entropy(actual,tags()[:,2]).backward()
        self.assertGreater(float(head.query.weight.grad.norm()),0)
        with torch.no_grad():head.query.weight.add_(-.1*head.query.weight.grad)
        head.zero_grad();c=make_context(layout,adapter(z))
        head(scores,q,tags()[:,:2],c).square().sum().backward()
        self.assertGreater(float(head.prototype.weight.grad.norm()),0)
        adapter.zero_grad();adapter(z).square().mean().backward()
        self.assertGreater(float(adapter.up.weight.grad.norm()),0)

    def test_sparse_context_matches_dense_reference_gradients_inverse_selfloop(self):
        t=tags();z=torch.randn(2*len(t),8,requires_grad=True);l=make_layout(t,6,2);c=make_context(l,z)
        for e in range(6):
            mask=l.ids==e
            if mask.any():
                selected=l.pair_entities==e
                weighted=(c.residuals[selected]*l.pair_counts[selected,None]).sum(0)
                torch.testing.assert_close(weighted,torch.zeros(8),atol=2e-6,rtol=0)
        h=ScoreResidual(8,2,2)
        with torch.no_grad():h.query.weight.normal_()
        q=torch.randn(len(t),8,requires_grad=True);base=torch.randn(len(t),6)
        actual=h(base,q,t[:,:2],c);expected=base.clone()
        for i in range(len(t)):
            for j,(e,r) in enumerate(zip(l.pair_entities.tolist(),l.pair_contexts.tolist())):
                if r==(int(t[i,1])+2)%4:
                    expected[i,e]+=(h.query(q[i])*h.prototype(c.residuals[j])).sum()/2**.5
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        ga=torch.autograd.grad(actual.sum(),(z,q),retain_graph=True)
        gb=torch.autograd.grad(expected.sum(),(z,q))
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
        # Entity4 has no replay and receives exactly zero correction.
        torch.testing.assert_close(actual[:,4],base[:,4],rtol=0,atol=0)
        for chunk in [1,3,100]:
            h.chunk_size=chunk;torch.testing.assert_close(h(base,q,t[:,:2],c),actual,rtol=1e-6,atol=1e-6)

    def test_shuffle_keeps_counts_entity_roles_and_global_rng(self):
        t=tags();state=torch.get_rng_state().clone();a=make_layout(t,6,2);b=make_layout(t,6,2,True,71)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(a.pair_entities,b.pair_entities);torch.testing.assert_close(a.pair_counts,b.pair_counts)
        torch.testing.assert_close(a.pair_contexts,b.pair_contexts)
        n=len(t)
        for role in range(2):
            for e in range(6):
                idx=torch.arange(role*n,(role+1)*n);idx=idx[a.ids[idx]==e]
                self.assertEqual(sorted(a.contexts[idx].tolist()),sorted(b.contexts[idx].tolist()))
        torch.testing.assert_close(b.contexts,make_layout(t,6,2,True,71).contexts)

    def test_empty_singleton_bad_ids_and_cli(self):
        empty=torch.empty(0,3,dtype=torch.long);c=make_context(make_layout(empty,6,2),torch.empty(0,8))
        h=ScoreResidual(8,2);scores=torch.randn(1,6)
        torch.testing.assert_close(h(scores,torch.randn(1,8),torch.tensor([[0,0]]),c),scores)
        t=torch.tensor([[0,3,0]]);c=make_context(make_layout(t,6,2),torch.randn(2,8))
        self.assertEqual(h(scores,torch.randn(1,8),t[:,:2],c).shape,(1,6))
        for bad in [torch.tensor([[6,0,1]]),torch.tensor([[0,4,1]]),torch.tensor([[-1,0,1]])]:
            with self.assertRaises(ValueError):make_layout(bad,6,2)
        with self.assertRaises(ValueError):h(scores,torch.randn(1,8),t,c)
        p=build_arg_parser();self.assertEqual(p.parse_args([]).rcpr_mode,'original')
        self.assertEqual(p.parse_args(['--rcpr']).rcpr_mode,'prototype')
        for mode in ('mean-shared','mean-head','prototype','shuffled'):self.assertEqual(args(mode).rcpr_mode,mode)

    def test_live_training_prediction_first_stage_empty_replay_and_reload(self):
        a=args('prototype');m=model(a);features=torch.randn(len(tags()),3,8)
        optimizer=torch.optim.Adam(m.parameters(),lr=.01)
        for replay in [None,features,torch.empty(0,3,8)]:
            m.train();optimizer.zero_grad();t=tags() if replay is not None else None
            losses=m.get_loss(history(),tags()[:1],None,False,model_output=replay,
                model_output_x_t=replay[:,2] if replay is not None else None,tag=t,diffuc='add',output_reply_triple=t)
            (.7*losses[0]+.3*losses[1]+losses[2]).backward();optimizer.step()
            self.assertIsNone(m._rcpr_context)
        self.assertGreater(float(m.rcpr_head.query.weight.norm()),0)
        m.eval();pred=m.predict(history(),2,None,tags()[:1],features,features[:,2],tags(),False)
        self.assertEqual(pred[1].shape,(2,6));self.assertIsNone(m._rcpr_context)
        clone=model(a);clone.load_state_dict(m.state_dict());clone.eval()
        other=clone.predict(history(),2,None,tags()[:1],features,features[:,2],tags(),False)
        torch.testing.assert_close(pred[1],other[1],rtol=0,atol=0)
        with torch.no_grad():clone.rcpr_head.query.weight.zero_()
        zero=clone.predict(history(),2,None,tags()[:1],features,features[:,2],tags(),False)
        self.assertGreater(float((zero[1]-pred[1]).abs().max()),0)

    def test_real_guidance_uses_head_input_gradient_and_preserves_terminal_rng(self):
        a=args('prototype');m=model(a).requires_grad_(False).train()
        with torch.no_grad():m.rcpr_head.query.weight.normal_(std=.1)
        with contextlib.redirect_stdout(io.StringIO()):
            g=Att_Diffuse_model(create_model_diffu(a),a,'uvrgcn',6,2,max_time=4,dropout=.2)
        g.requires_grad_(False).train();calls=[]
        hook=m.rcpr_head.register_forward_hook(lambda *x:calls.append(True))
        with torch.no_grad():output=g(None,tags(),a,False,False,ct=2,model=m,history_glist=history())[1]
        hook.remove();self.assertTrue(calls);self.assertEqual(output.shape,(len(tags()),3,8))
        self.assertTrue(torch.isfinite(output).all());self.assertIsNone(m._rcpr_context)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        # Same Original terminal draw is consumed when it is replaced by mean.
        d=g.diffu;t=torch.tensor([3]*len(tags()));x=torch.randn(len(tags()),8);x0=torch.randn(len(tags()),3,8)
        mean=torch.randn_like(x);var=torch.zeros(len(tags()),1)
        with patch.object(d,'p_mean_variance',return_value=(mean,var,x0)):
            state=rng_state('cpu');a.rcpr_mode='original'
            d.p_sample(None,x,m,tags(),t,None,None,None,None,None,None,False,a);expected_rng=torch.get_rng_state()
            restore_rng(state,'cpu');a.rcpr_mode='prototype'
            result=d.p_sample(None,x,m,tags(),t,None,None,None,None,None,None,False,a)
            torch.testing.assert_close(result[:,2],mean,rtol=0,atol=0)
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))

    def test_mean_shared_is_original_occurrence_adapter_and_no_extra_decoder_calls(self):
        a=args('mean-shared');m=model(a);f=torch.randn(len(tags()),3,8);calls=[]
        with torch.no_grad():m.rcpr_adapter.up.weight.normal_(std=.1)
        ids,bank=m.select_entity(tags(),f,False)
        z=m.rcpr_adapter(torch.cat((f[:,0],f[:,2])));occ=torch.cat((tags()[:,0],tags()[:,2]))
        for e in ids:torch.testing.assert_close(bank[e],z[occ==e].mean(0),rtol=0,atol=0)
        m.eval();hook=m.decoder_ob.bn0.register_forward_hook(lambda *x:calls.append(1))
        m.predict(history(),2,None,tags()[:1],f,f[:,2],tags(),False);hook.remove()
        self.assertEqual(len(calls),1);self.assertIsNone(m.rcpr_head)

    def test_integer_layout_cache_has_no_differentiable_features(self):
        m=model(args('prototype'));t=tags();f=torch.randn(len(t),3,8,requires_grad=True)
        m.get_loss(history(),t[:1],None,False,model_output=f,model_output_x_t=f[:,2],tag=t,
                   diffuc='add',output_reply_triple=t)
        self.assertIsNone(m._rcpr_context)
        self.assertTrue(all(not x.requires_grad for x in vars(m._rcpr_layout).values() if torch.is_tensor(x)))
        layout=m._rcpr_layout;m.rcpr_layout(t);self.assertIs(layout,m._rcpr_layout)
        t[0,1]=1;m.rcpr_layout(t);self.assertIsNot(layout,m._rcpr_layout)
        layout=m._rcpr_layout;m.rcpr_layout(t.clone());self.assertIsNot(layout,m._rcpr_layout)

    def test_probe_fit_uses_train_targets_and_gate_requires_controls(self):
        t=tags();c=make_context(make_layout(t,6,2),torch.randn(2*len(t),8))
        q=torch.randn(len(t),8);base=torch.randn(len(t),6);state=torch.get_rng_state().clone()
        head,report=fit_head(q,base,t[:,:2],t[:,2],c,2,2,.01)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.isfinite(head.query.weight).all());self.assertEqual(report['steps'],2)
        self.assertFalse(gate([])['pass'])
        m=lambda value:dict(mrr=value,forward={'mrr':value},inverse={'mrr':value})
        records=[{'metrics':{name:m(.51 if name=='prototype' else .50) for name in ('mean-shared','mean-head','prototype','shuffled')}} for _ in range(5)]
        self.assertTrue(gate(records)['pass']);records[0]['metrics']['shuffled']['mrr']=.6
        self.assertFalse(gate(records)['pass'])

    def test_frozen_diagnostic_main_on_synthetic_data(self):
        import sys
        path=ROOT/'scripts/diagnose_rcpr.py'
        spec=importlib.util.spec_from_file_location('rcpr_diagnostic_smoke',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        a=args('mean-shared');m=model(a).requires_grad_(False).eval()
        with contextlib.redirect_stdout(io.StringIO()):
            generator=Att_Diffuse_model(create_model_diffu(a),a,'uvrgcn',6,2,max_time=5)
        generator.requires_grad_(False).train()
        data=[np.array([[0,0,1],[0,1,1],[2,1,3],[1,0,5]]) for _ in range(5)]
        validation=[np.array([[0,0,1],[2,1,3]]) for _ in range(5)]
        f=torch.randn(len(tags()),3,8)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'probe'
            argv=['diagnose_rcpr.py','--run-dir',str(Path(tmp)/'source'),'--output',str(out),
                  '--gpu','-1','--stages','2,3,4','--facts','2','--steps','1','--rank','2']
            with patch.object(sys,'argv',argv),patch.object(module,'load',return_value=(a,{'synthetic':True},
                data,validation,None,m,generator,torch.device('cpu'))),patch.object(generator,'gereration_feature',
                return_value=(torch.tensor(data[0]),f,tags())),contextlib.redirect_stdout(io.StringIO()):
                module.main()
            report=json.loads((out/'summary.json').read_text())
            self.assertEqual(report['test_queries_scored'],0);self.assertEqual(report['backbone_updates'],0)
            self.assertEqual(len(report['stages']),3)
            self.assertEqual(set(report['stages'][0]['metrics']),{'mean-shared','mean-head','prototype','shuffled'})
            self.assertTrue((out/'queries.csv').exists())
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_all_head_modes_and_static_graph_have_live_scores(self):
        for mode in ['mean-head','shuffled','prototype']:
            a=args(mode);m=model(a);f=torch.randn(len(tags()),3,8)
            m.eval()
            with torch.no_grad():m.rcpr_head.query.weight.normal_(std=.1)
            out=m.predict(history(),2,None,tags()[:1],f,f[:,2],tags(),False)
            self.assertTrue(torch.isfinite(out[1]).all());self.assertEqual(out[1].shape,(2,6))
        # Actual static encoder with an extra word node, not just static=False.
        a=args('prototype');a.add_static_graph=True
        with contextlib.redirect_stdout(io.StringIO()):
            m=RecurrentRGCN('convtranse','uvrgcn',6,2,1,1,8,2,'sub',sequence_len=3,
                num_bases=2,num_hidden_layers=1,self_loop=True,layer_norm=True,use_static=True,
                entity_prediction=True,relation_prediction=True,args=a)
        graph=build_sub_graph(7,1,np.array([[0,0,6],[1,0,6],[2,0,6]]),False,-1)
        losses=m.get_loss(history(),tags()[:1],graph,False,model_output=f,model_output_x_t=f[:,2],
            tag=tags(),diffuc='add',output_reply_triple=tags())
        loss=.7*losses[0]+.3*losses[1]+losses[2];loss.backward()
        self.assertTrue(torch.isfinite(loss).all());self.assertIsNotNone(m.rcpr_head.query.weight.grad)

    def test_checkpoint_optimizer_resume_keeps_next_update_identical(self):
        a=args('prototype');m=model(a);d=torch.nn.Linear(2,2)
        opt=torch.optim.Adam(m.parameters(),lr=.01);od=torch.optim.Adam(d.parameters(),lr=.01)
        f=torch.randn(len(tags()),3,8)
        def step(m,opt):
            m.train();opt.zero_grad();out=m.get_loss(history(),tags()[:1],None,False,
                model_output=f,model_output_x_t=f[:,2],tag=tags(),diffuc='add',output_reply_triple=tags())
            (.7*out[0]+.3*out[1]+out[2]).backward();opt.step()
        step(m,opt)
        with tempfile.TemporaryDirectory() as tmp:
            a.run_dir=Path(tmp);a._provenance={'protocol':'rcpr-prototype-v1','synthetic':True}
            save_training_state(a,2,m,d,opt,od,'cpu',None);state=rng_state('cpu');step(m,opt)
            clone=model(a);teacher=model(a);other=torch.nn.Linear(2,2);dt=copy.deepcopy(other)
            oc=torch.optim.Adam(clone.parameters(),lr=.01);do=torch.optim.Adam(other.parameters(),lr=.01)
            a.resume=a.run_dir/'latest.pt'
            start,_=restore_training_state(a,clone,other,teacher,dt,oc,do,'cpu');self.assertEqual(start,3)
            step(clone,oc);self.assert_tensors(m.state_dict(),clone.state_dict())
            self.assertIsNone(clone._rcpr_context)

    def test_passive_records_do_not_change_rng_or_weights(self):
        a=args('prototype');m=model(a);f=torch.randn(len(tags()),3,8);state=rng_state('cpu');before=copy.deepcopy(m.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            a.run_dir=Path(tmp);a._provenance={'protocol':'rcpr-prototype-v1'}
            record_training_stage(a,1,m,f,tags())
            rec=EvaluationRecorder(a,'validation',[np.array([[0,0,1]])]*3,2)
            rec.record(1,tags()[:2],torch.tensor([2,3]),torch.tensor([1,2]),m,f)
            rec.record(2,tags()[:2],torch.tensor([2,3]),torch.tensor([1,2]),m,f);rec.finish(2)
            s=json.loads((rec.directory/'summary.json').read_text());self.assertEqual(s['macro_mrr'],.75)
        self.assert_tensors(before,m.state_dict());self.assertTrue(torch.equal(state['torch'],torch.get_rng_state()))


if __name__=='__main__':unittest.main()
