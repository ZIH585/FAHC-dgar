"""Frozen-checkpoint VALIDATION intervention: noise removal versus tail scale.

Four arms share one actual denoiser/guide forward, one HCP/noise realization and
identical head/relation tokens. No training, optimizer or target-dependent scale.
Run separately on Original and mean checkpoints; compare arms within each run.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from diffusion import dcr
from rgcn import utils
from src.runtime import fingerprint
from scripts.diagnose_dcr import (load_original, stage_state, evaluation_history, metrics)

ARMS = ('original', 'mean', 'scaled-original', 'scaled-mean')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed checkpoint, read-only')
    p.add_argument('--checkpoint-mode', choices=('original', 'mean'), default='original')
    p.add_argument('--output', type=Path, required=True, help='Fresh output outside input run')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=1, help='Diagnostic noise seed, not new training seed')
    p.add_argument('--stages', default='30,60,90,120,150,180,210,240,270,300,330')
    p.add_argument('--check-only', action='store_true', help='Load only, stop before inference')
    return p


@torch.no_grad()
def scale_arms(original, mean_tail):
    if original.ndim != 3 or original.shape[1] != 3 or mean_tail.shape != original[:, -1].shape:
        raise ValueError('Expected features [facts,3,hidden] and mean [facts,hidden]')
    if not original.is_floating_point() or mean_tail.dtype != original.dtype or mean_tail.device != original.device:
        raise ValueError('Feature/mean dtype and device must match')
    x = original[:, -1].detach()
    m = mean_tail.detach()
    nx, nm = x.norm(dim=-1, keepdim=True), m.norm(dim=-1, keepdim=True)
    tiny = max(1e-12, torch.finfo(x.dtype).tiny)
    variants = {}
    for name, tail in (('original', x), ('mean', m),
                       ('scaled-original', x * (nm/nx.clamp_min(tiny))),
                       ('scaled-mean', m * (nx/nm.clamp_min(tiny)))):
        variants[name] = torch.cat((original[:, :2].detach(), tail.unsqueeze(1)), dim=1).detach()
    if len(x):
        facts = len(x)
        nonzero = (nx[:, 0] > tiny) & (nm[:, 0] > tiny)
        scale_stats = dict(facts=facts, zero_original_tails=int((nx<=tiny).sum().cpu()),
            zero_mean_tails=int((nm<=tiny).sum().cpu()),
            norm_means={name:float(v[:, -1].norm(dim=-1).mean().cpu()) for name,v in variants.items()},
            max_norm_mismatch={name:float((variants[name][:, -1].norm(dim=-1)[nonzero]-target[:, 0][nonzero]).abs().max().cpu())
                if nonzero.any() else None for name,target in (('scaled-original',nm),('scaled-mean',nx))})
    else:
        scale_stats = dict(facts=0, zero_original_tails=0, zero_mean_tails=0, norm_means={}, max_norm_mismatch={})
    return variants, scale_stats


@torch.no_grad()
def paired_tails(generator, model, args, tags, guide, static, stage, cuda):
    if args.vl or args.delete_score:
        raise ValueError('Requires the original guided replay sampling path')
    if not len(tags):
        return {name:None for name in ARMS}, dict(facts=0, denoiser_calls=0)
    # A mean-trained checkpoint still needs x=m+noise for its matched control.
    # Reset only this local sampler argument; never change checkpoint/config.
    sampling_args = copy.copy(args)
    sampling_args.dcr_mode = 'original'
    sampling_args._dcr_recorder = None
    sampler, captured = generator.diffu, {}
    original_mean = sampler.p_mean_variance

    def capture(*a, **kw):
        if captured:
            raise ValueError('Expected one denoiser/guide call')
        result = original_mean(*a, **kw)
        captured['mean'] = result[0].detach().clone()
        return result

    had_override = 'p_mean_variance' in sampler.__dict__
    old_override = sampler.__dict__.get('p_mean_variance')
    sampler.p_mean_variance = capture
    try:
        original = generator(None, tags, sampling_args, False, cuda, ct=stage,
            model=model, history_glist=guide, static_graph=static, triples=None)[1].detach()
    finally:
        if had_override:
            sampler.p_mean_variance = old_override
        else:
            del sampler.p_mean_variance
        dcr.release_guidance_graph(model)
    variants, stats = scale_arms(original, captured['mean'])
    stats.update(denoiser_calls=1, posterior_x0_coefficient=float(sampler.posterior_mean_coef1[-1]))
    return variants, stats


def summarize(rows):
    overall = {a:metrics(rows,a+'_rank') for a in ARMS}
    by_stage = {str(t):{a:metrics([r for r in rows if r['stage']==t],a+'_rank') for a in ARMS}
                for t in sorted({r['stage'] for r in rows})}
    contrasts = {}
    for method,control in (('mean','original'),('scaled-original','original'),
                           ('mean','scaled-original'),('scaled-mean','original')):
        deltas = [1/r[method+'_rank']-1/r[control+'_rank'] for r in rows]
        positive = sorted((d for d in deltas if d>0),reverse=True)
        contrasts[method+' vs '+control] = dict(pooled_delta_pp=100*sum(deltas)/len(rows),
            macro_delta_pp=100*np.mean([v[method]['mrr']-v[control]['mrr'] for v in by_stage.values()]),
            improved=sum(d>0 for d in deltas),worsened=sum(d<0 for d in deltas),unchanged=sum(d==0 for d in deltas),
            positive_stages=sum(v[method]['mrr']>v[control]['mrr'] for v in by_stage.values()),
            net_rr_gain=float(sum(deltas)),top5_positive_rr_gain=float(sum(positive[:5])))
    return dict(overall=overall,by_stage=by_stage,contrasts=contrasts,
        by_direction={d:{a:metrics([r for r in rows if r['direction']==d],a+'_rank') for a in ARMS}
                      for d in ('forward','inverse')})


def run(options):
    if options.gpu < -1 or options.seed < 0:
        raise ValueError('Invalid GPU/seed')
    stages = [int(x) for x in options.stages.split(',')]
    if not stages or stages!=sorted(set(stages)) or min(stages)<2:
        raise ValueError('Stages must increase uniquely and be >=2')
    output,source=options.output.resolve(),options.run_dir.resolve()
    if output==source or source in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError('Use a fresh output outside the checkpoint directory')
    args,config,source_hash,train,valid,static,model,gen,device,cuda=load_original(options)
    if max(stages)>=len(train):
        raise ValueError('Stage outside dataset')
    if options.check_only:
        print(f'Validated {options.checkpoint_mode} checkpoint: {len(train)} stages; stopped before inference/training.')
        return
    output.mkdir(parents=True,exist_ok=True)
    answers=utils.load_all_answers_for_time_filter(valid,args.num_rels,args.num_nodes,False)
    rows,generation=[],[]
    for t in stages:
        if not len(valid[t]):
            continue
        rng=random.Random(options.seed+1000003*t)
        times=list(range(t)) if t-1<args.reply_batch else rng.sample(list(range(t)),args.reply_batch)
        noise_seed=rng.getrandbits(63)
        facts=set()
        for i in times:
            with (args.data_root/args.dataset/'history_snap_v2'/f'snap_{i}.pkl').open('rb') as stream:
                cache=pickle.load(stream)
            for s in set(map(int,valid[t][:,0])):
                facts.update(tuple(map(int,x)) for x in cache.get(str(s),()))
        tags=torch.as_tensor(list(facts),device=device,dtype=torch.long).reshape(-1,3)
        queries=torch.as_tensor(valid[t],device=device,dtype=torch.long)
        graphs=[utils.build_sub_graph(args.num_nodes,args.num_rels,x,cuda,args.gpu) for x in evaluation_history(train,t,args)]
        guide=[utils.build_sub_graph(args.num_nodes,args.num_rels,x,cuda,args.gpu) for x in evaluation_history(train,t-1,args)]
        start=time.monotonic()
        with stage_state(model,gen,device,noise_seed):
            variants,stats=paired_tails(gen,model,args,tags,guide,static,t,cuda)
            ranks,bank_norms={},{}
            with torch.no_grad():
                for arm,features in variants.items():
                    expanded,scores,_=model.predict(graphs,args.num_rels,static,queries,features,
                        features[:,-1] if features is not None else None,tags,cuda)
                    ranks[arm]=utils.get_total_rank(expanded,scores.clone(),answers[t],eval_bz=1000)[3].cpu().tolist()
                    if features is not None:
                        ids,bank=model.select_entity(tags,features,cuda)
                        bank_norms[arm]=float(bank[ids].norm(dim=-1).mean().cpu())
                    else:
                        bank_norms[arm]=None
                    del scores
        for i,(s,r,o) in enumerate(expanded.cpu().tolist()):
            rows.append(dict(stage=t,subject=s,relation=r,target=o,direction='forward' if i<len(queries) else 'inverse',
                **{a+'_rank':int(ranks[a][i]) for a in ARMS}))
        stats.update(stage=t,times=times,noise_seed=noise_seed,aggregate_entity_norm=bank_norms,seconds=time.monotonic()-start)
        generation.append(stats)
        print(f'Stage {t}: '+', '.join(f'{a} MRR={metrics([r for r in rows if r["stage"]==t],a+"_rank")["mrr"]:.6f}' for a in ARMS),flush=True)
        del variants
    if not rows:
        raise ValueError('No validation queries')
    summary=summarize(rows)
    summary.update(protocol='paired-dcr-scale-validation-v1',training_provenance=config,
        core_source_sha256=source_hash,
        diagnostic_sha256=fingerprint([Path(__file__),ROOT/'scripts/diagnose_dcr.py'],ROOT),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},generation=generation,
        limits='Frozen final model. Selected FULL VALIDATION slices; per-stage diagnostic RNG/buffers isolated, not paper Average or full author evaluation trajectory. One HCP/noise realization per stage, one denoiser forward shared by all arms. Per-fact tail norm matching does not ensure matching aggregated entity norms or eliminate all training-distribution effects. Zero tail cases cannot match a nonzero target norm. No optimization; targets used only after prediction for ranks.')
    with (output/'queries.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'generation.jsonl').open('w') as stream:
        for item in generation:
            stream.write(json.dumps(item,sort_keys=True)+'\n')
    print(json.dumps(summary['contrasts'],indent=2),flush=True)
    print(f'Completed without training. Outputs: {output}',flush=True)


if __name__=='__main__':
    run(parser().parse_args())
