"""Small posthoc RCPR screen using an audited completed mean+Shared checkpoint.

Only fresh score heads learn from TRAIN. Features are generated ONCE per stage
using Original HCP filtering by the current TRAIN snapshot's subjects. Controls
share this exact bank, backbone, query output, candidate scores and labels.
The final frozen backbone has seen later training snapshots: this is NOT an
online CL result or the author's retrospective final evaluation protocol.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.rcpr import make_layout, make_context, context_statistics
from src.rcpr_probe import fit_head, gate
from src.rrgcn import RecurrentRGCN
from src.runtime import UPSTREAM_COMMIT, fingerprint, validate_history_cache

PGR_SOURCE = '5e56dd4ae17f5ff08a0ad072f1bfeb2187f840f75adaf5d73bb0922bc63a481c'


def migrate_weights(model, weights):
    """Explicit read-only migration of the audited mean+Shared source only."""
    mapping = {f'pgr_adapter.{factor}.weight': f'rcpr_adapter.{factor}.weight' for factor in ('up', 'down')}
    if not set(mapping) <= weights.keys():
        raise ValueError('Missing mean+Shared adapter weights')
    migrated = {mapping.get(key, key): value for key, value in weights.items()}
    expected = model.state_dict()
    if migrated.keys() != expected.keys():
        raise ValueError('Mean+Shared keys differ; only the two adapter key renames are allowed')
    expected.update(migrated)
    model.load_state_dict(expected, strict=True)


def load(options):
    directory = options.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    saved = config['settings']
    if (config['protocol'] != 'pgr-mean-shared-v1' or config['upstream_commit'] != UPSTREAM_COMMIT
            or config['source_sha256'] != PGR_SOURCE or saved.get('pgr_mode') != 'mean-shared'):
        raise ValueError('Expected the audited, completed PGR mean+Shared checkpoint')
    args = build_arg_parser().parse_args([])
    for key, value in saved.items():
        if not key.startswith('pgr_'):
            if not hasattr(args, key):
                raise ValueError(f'Unknown scientific setting: {key}')
            setattr(args, key, value)
    args.rcpr_mode, args.rcpr_adapter_rank = 'mean-shared', saved['pgr_adapter_rank']
    args.rcpr_rank = options.rank
    args.rcpr_diagnostics = False
    args.gpu, args.data_root, args.run_dir = options.gpu, options.data_root.resolve(), directory
    validate_args(args)
    torch.default_generator.manual_seed(args.seed if args.seed is not None else 0)
    data_dir = args.data_root / args.dataset
    paths = [data_dir / f'{name}.txt' for name in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        paths.append(data_dir / 'e-w-graph.txt')
    if (fingerprint(paths, data_dir) != config['data_sha256']
            or fingerprint(list((data_dir / 'history_snap_v2').glob('snap_*.pkl')), data_dir) != config['history_sha256']):
        raise ValueError('Dataset or HCP cache differ from the source run')
    latest = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    if latest.get('version') != 2 or latest['provenance'] != config:
        raise ValueError('Checkpoint provenance/version mismatch')
    data = utils.load_data(args.dataset, data_root=args.data_root)
    slices = sum((utils.split_by_time(rows) for rows in (data.train, data.valid, data.test)), [])
    train, validation, _ = utils.spilt_data(slices)
    if latest['completed_stage'] != len(train) - 1:
        raise ValueError('Source run is incomplete')
    validate_history_cache(train, data.num_rels, data_dir / 'history_snap_v2')
    args.num_nodes, args.num_rels = data.num_nodes, data.num_rels
    if options.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; --gpu -1 is explicit CPU mode')
    device = torch.device(f'cuda:{options.gpu}' if options.gpu >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    words, static_relations, static = 0, 0, None
    if args.add_static_graph:
        triples = np.asarray(_read_triplets_as_list(str(data_dir / 'e-w-graph.txt'), {}, {}, load_time=False))
        static_relations, words = len(np.unique(triples[:, 1])), len(np.unique(triples[:, 2]))
        triples[:, 2] += data.num_nodes
        static = utils.build_sub_graph(data.num_nodes + words, static_relations, triples, device.type == 'cuda', args.gpu)
    model = RecurrentRGCN(args.decoder, args.encoder, data.num_nodes, data.num_rels, static_relations,
        words, args.n_hidden, args.attn_heads, args.opn, sequence_len=args.train_history_len,
        num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
        layer_norm=args.layer_norm, input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout, aggregation=args.aggregation, weight=args.weight,
        discount=args.discount, angle=args.angle, use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction, relation_prediction=args.relation_prediction,
        use_cuda=device.type == 'cuda', gpu=args.gpu, args=args)
    generator = Att_Diffuse_model(create_model_diffu(args), args, args.encoder, data.num_nodes,
        data.num_rels, dropout=args.dropout, max_time=len(train), num_words=words,
        num_static_rels=static_relations, num_bases=args.n_bases)
    for filename, key in (('model.pt', 'model'), ('diffusion.pt', 'diffusion')):
        weights = torch.load(directory / filename, map_location='cpu', weights_only=True)
        if weights.keys() != latest[key].keys() or any(not torch.equal(weights[k], latest[key][k]) for k in weights):
            raise ValueError(f'{filename} differs from completed latest.pt')
        if key == 'model':
            migrate_weights(model, weights)
        else:
            generator.load_state_dict(weights, strict=True)
    model.to(device).requires_grad_(False).eval()
    generator.to(device).requires_grad_(False).train()
    return args, config, train, validation, static, model, generator, device


def pick(rows, limit, seed, device):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=generator)[:limit].numpy()
    if torch.is_tensor(rows):
        return rows.index_select(0, torch.as_tensor(indices,device=rows.device)).to(device)
    return torch.as_tensor(rows[indices], dtype=torch.long, device=device)



def ranks(triples, scores, answers):
    # The same integer ranking/filtering routine used in Original DGAR.
    _, _, raw, filtered = utils.get_total_rank(triples, scores.clone(), answers, eval_bz=1000, rel_predict=0)
    return raw.detach().cpu().tolist(), filtered.detach().cpu().tolist()


def metrics(values):
    return {'queries': len(values), 'mrr': sum(1/x for x in values)/len(values),
            **{f'hits{k}':sum(x<=k for x in values)/len(values) for k in (1,3,10)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Audited PGR mean+Shared source run, read only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stages', default='60,120,180,240,300')
    parser.add_argument('--facts', type=int, default=128, help='Probe TRAIN rows per current/past source, not an HCP budget')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--score-chunk', type=int, default=256)
    parser.add_argument('--check-only', action='store_true', help='Validate/load only; no model forward/update')
    options = parser.parse_args()
    try:
        stages = [int(value) for value in options.stages.split(',')]
    except ValueError:
        parser.error('stages must be comma-separated integers')
    if (stages != sorted(set(stages)) or not stages or stages[0] < 2 or options.facts < 1
            or options.steps < 1 or options.rank < 1 or options.score_chunk < 1
            or not math.isfinite(options.lr) or options.lr <= 0):
        parser.error('invalid stage order, facts, rank, steps, chunk size or learning rate')
    output = options.output.resolve()
    if output == options.run_dir.resolve() or options.run_dir.resolve() in output.parents or output.exists():
        parser.error('output must be a new directory outside the source run')
    args, config, train, validation, static, model, generator, device = load(options)
    if stages[-1] >= len(train):
        parser.error('stage outside source sequence')
    if options.check_only:
        print('Validated source weights, dataset, HCP cache and migration. No forward/update performed.')
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps({'source':config,'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        'probe_source_sha256':fingerprint([ROOT/'scripts/diagnose_rcpr.py', ROOT/'src/rcpr_probe.py', ROOT/'src/rcpr.py'],ROOT)},indent=2)+'\n')
    seed = args.seed if args.seed is not None else 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    answers = utils.load_all_answers_for_time_filter(validation,args.num_rels,args.num_nodes,False)
    rows,records = [],[]
    for stage in stages:
        started=time.monotonic()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        history=[utils.build_sub_graph(args.num_nodes,args.num_rels,s,device.type=='cuda',args.gpu)
                 for s in train[max(0,stage-args.train_history_len):stage]]
        previous=[utils.build_sub_graph(args.num_nodes,args.num_rels,s,device.type=='cuda',args.gpu)
                  for s in train[max(0,stage-1-args.train_history_len):stage-1]]
        current=pick(train[stage],options.facts,seed+stage,device)
        valid=torch.as_tensor(validation[stage],dtype=torch.long,device=device)
        with torch.no_grad():
            _,features,tags=generator.gereration_feature(args,stage,generator,model,previous,None,static,
                train[stage],current,device.type=='cuda',train)
            if features is None or not len(features) or not len(valid):
                records.append({'stage':stage,'reason':'no_replay_or_validation'})
                continue
            # TRAIN HCP contains no future/held-out labels. Keep all HCP facts for
            # features; only limit which historical labels train the tiny probe.
            past=tags.clone()
            mask=past[:,1]>=args.num_rels
            past[mask]=past[mask][:,[2,1,0]];past[mask,1]-=args.num_rels
            past=pick(past,options.facts,seed+stage+1,device)
            fit_queries=model.get_all_triple(torch.cat((current,past)))
            valid_queries=model.get_all_triple(valid)
            query_triples=torch.cat((fit_queries,valid_queries))
            states,_,relations,_,_=model.forward(history,static,device.type=='cuda',features,features[:,2],tags,diffu_rep='add')
            embedding=F.normalize(states[-1]) if model.layer_norm else states[-1]
            base,query=model.decoder_ob(embedding,relations,query_triples,mode='test',return_query=True)
            occurrences=model.rcpr_adapter(torch.cat((features[:,0],features[:,2])))
            _,bank=model.select_entity(tags,features,device.type=='cuda')
            contexts={name:make_context(make_layout(tags,args.num_nodes,args.num_rels,shuffled=name=='shuffled',
                        seed=seed+args.rcpr_shuffle_seed),occurrences,bank).detached()
                      for name in ('prototype','shuffled')}
            model.clear_rcpr_context()
        n=len(fit_queries)
        stage_rows=[];record={'stage':stage,'metrics':{},'fitting':{},
            'context':{k:context_statistics(v,occurrences) for k,v in contexts.items()}}
        repeated=set()
        for snap in train[:stage]:
            for s,r,o in snap:repeated.update(((int(s),int(r),int(o)),(int(o),int(r)+args.num_rels,int(s))))
        labels=valid_queries.cpu().tolist()
        for mode in ('mean-shared','mean-head','prototype','shuffled'):
            if mode=='mean-shared':
                scores=base[n:]
            else:
                context=contexts['shuffled' if mode=='shuffled' else 'prototype']
                head,fit=fit_head(query[:n],base[:n],fit_queries[:,:2],fit_queries[:,2],context,
                    options.rank,options.steps,options.lr,mean_control=mode=='mean-head',chunk=options.score_chunk)
                with torch.no_grad():scores=head(base[n:],query[n:],valid_queries[:,:2],context,mode=='mean-head')
                record['fitting'][mode]=fit
            raw,filtered=ranks(valid_queries,scores,answers[stage])
            m=metrics(filtered)
            for name,flag in [('forward',False),('inverse',True)]:
                m[name]=metrics([rank for (_,r,_),rank in zip(labels,filtered) if (r>=args.num_rels)==flag])
            record['metrics'][mode]=m
            for (s,r,o),a,b in zip(labels,raw,filtered):
                stage_rows.append(dict(stage=stage,mode=mode,subject=s,relation=r,target=o,inverse=int(r>=args.num_rels),
                    seen_before=int((s,r,o) in repeated),rank_raw=a,rank_filtered=b))
        record['seconds']=time.monotonic()-started
        record['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        records.append(record);rows.extend(stage_rows)
        with (output/'stages.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    summary={'kind':'posthoc_frozen_checkpoint_probe_not_online_or_final_MRR','source':config,
        'stages':records,'gate':gate(records),'test_queries_scored':0,'backbone_updates':0,
        'features':'one Original TRAIN-subject HCP bank per stage, shared by all controls',
        'sampling':'terminal mean only; upstream randomness retained'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if rows:
        with (output/'queries.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('RCPR frozen diagnostic:',output,flush=True)
    print('Prespecified gate:',json.dumps(summary['gate'],allow_nan=False),flush=True)


if __name__=='__main__':main()
