"""Paired validation-only context aggregation diagnostic; never trains a model."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import PROTOCOL, UPSTREAM_COMMIT, fingerprint, validate_history_cache

ACCEPTED_ORIGINAL_SOURCES = {
    # codex/original-baseline@23dd43d / bedb5b1
    '6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538',
    # codex/grb@caf2193, accepted ONLY with mode=original; CPU equivalence checked.
    'c22dc00022ff00ef8ebf9486d32f839ef9f60b5e0c59d04b79413401ba96b3d5',
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed Original run; read-only')
    p.add_argument('--output', type=Path, required=True, help='New diagnostic directory')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--stages', default='60,120,180,240,300')
    p.add_argument('--facts-per-stage', type=int, default=12, help='Adds inverse queries, so 24 queries by default')
    p.add_argument('--seed', type=int, default=1, help='Diagnostic sampling seed; no training')
    p.add_argument('--check-only', action='store_true', help='Validate data, checkpoint and model shapes, then exit before inference')
    return p


def validate_config(config):
    if config['protocol'] != PROTOCOL or config['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Use the restored official-dgar-v1 Original checkpoint')
    if config['source_sha256'] not in ACCEPTED_ORIGINAL_SOURCES:
        raise ValueError('Unknown training source: use baseline@23dd43d or grb@caf2193 Original')
    s = config['settings']
    if s.get('grb_mode', 'original') != 'original':
        raise ValueError('Half/adaptive weights are not Original checkpoints')
    if any(k.startswith(('fahc', 'ecri', 'qcdr')) for k in s):
        raise ValueError('Previous experimental checkpoints are unsupported')
    if not s['diffuc'] or s['delete_feature_reply']:
        raise ValueError('This diagnostic requires Original generated feature replay')


def pool_for_query(tags, features, subject, relation, num_relations, seed):
    """Only known query (s,r) and historical replay enter this selector, never y.

    A tail occurrence of (a,r,b) has role inverse(r), including when the cached
    triple itself is inverse. Original mixes head/tail occurrences of an entity.
    """
    inverse = torch.where(tags[:, 1] < num_relations,
                          tags[:, 1]+num_relations, tags[:, 1]-num_relations)
    entities = torch.cat((tags[:, 0], tags[:, 2]))
    roles = torch.cat((tags[:, 1], inverse))
    vectors = torch.cat((features[:, 0], features[:, 2]))
    eligible = torch.nonzero(entities == subject, as_tuple=False).flatten()
    matched = eligible[roles[eligible] == relation]
    count = len(matched)
    if not count:
        return len(eligible), 0, None, None
    matching = vectors[matched].mean(0)
    if count == len(eligible):
        return count, count, matching, matching
    rng = random.Random(seed)
    selected = rng.sample(range(len(eligible)), count)
    indices = torch.as_tensor(selected, device=tags.device, dtype=torch.long)
    control = eligible[indices]
    return len(eligible), count, matching, vectors[control].mean(0)


@contextmanager
def selector_override(model, entities, bank, subject=None, vector=None):
    original = model.select_entity
    replacement = bank
    if vector is not None:
        replacement = bank.clone()
        replacement[subject] = vector
    model.select_entity = lambda *a, **kw: (entities, replacement)
    try:
        yield
    finally:
        model.select_entity = original


@torch.no_grad()
def entity_scores(model, history, static, queries, features, tags, use_cuda):
    states, _, relations, _, _ = model.forward(history, static, use_cuda,
        model_output=features, model_output_x_t=features[:, -1] if features is not None else None,
        tag=tags, diffu_rep=model.args.diffuc)
    h = F.normalize(states[-1]) if model.layer_norm else states[-1]
    # Keep the SAME full query batch in every arm. ConvTransE's singleton path
    # skips bn2 even in eval; decoding just one row would change more than pooling.
    return model.decoder_ob(h, relations, queries, mode='test')


def filtered_rank(query, score, answers):
    return int(utils.get_total_rank(query.reshape(1, 3), score.reshape(1, -1).clone(),
                                   answers, eval_bz=1)[3][0])


def metrics(rows, column):
    if not rows:
        return None
    ranks = np.asarray([r[column] for r in rows], dtype=float)
    return {'queries': len(rows), 'mrr': float((1/ranks).mean()),
            'hits1': float((ranks <= 1).mean()), 'hits3': float((ranks <= 3).mean()),
            'hits10': float((ranks <= 10).mean())}


def make_models(args, nodes, relations, words, static_relations, snapshots):
    model = RecurrentRGCN(args.decoder, args.encoder, nodes, relations, static_relations,
        words, args.n_hidden, args.attn_heads, args.opn, sequence_len=args.train_history_len,
        num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
        layer_norm=args.layer_norm, input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
        aggregation=args.aggregation, weight=args.weight, discount=args.discount,
        angle=args.angle, use_static=args.add_static_graph, entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction, use_cuda=args.gpu >= 0, gpu=args.gpu, args=args)
    generator = Att_Diffuse_model(create_model_diffu(args), args, args.encoder, nodes, relations,
        dropout=args.dropout, max_time=snapshots, num_words=words,
        num_static_rels=static_relations, num_bases=args.n_bases)
    return model, generator


def run(options):
    if options.gpu < -1 or options.facts_per_stage < 1 or options.seed < 0:
        raise ValueError('Invalid GPU, fact count or diagnostic seed')
    stages = [int(x) for x in options.stages.split(',')]
    if stages != sorted(set(stages)) or any(t < 2 for t in stages):
        raise ValueError('Stages must be unique, increasing, and >=2')
    output, run_dir = options.output.resolve(), options.run_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    if output == run_dir or run_dir in output.parents:
        raise ValueError('Keep diagnostic outputs outside the checkpoint directory')
    config = json.loads((run_dir/'config.json').read_text())
    validate_config(config)
    args = build_arg_parser().parse_args([])
    for key, value in config['settings'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.data_root, args.gpu = options.data_root.resolve(), options.gpu
    args = validate_args(args)
    use_cuda = args.gpu >= 0
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; --gpu -1 explicitly selects CPU')
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    data_dir = args.data_root/args.dataset
    data_files = [data_dir/f'{name}.txt' for name in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        data_files.append(data_dir/'e-w-graph.txt')
    if fingerprint(data_files, data_dir) != config['data_sha256']:
        raise ValueError('Data do not match the Original run')
    if fingerprint(list((data_dir/'history_snap_v2').glob('snap_*.pkl')), data_dir) != config['history_sha256']:
        raise ValueError('Historical cache does not match the Original run')
    local_source = fingerprint([p for folder in ('src', 'diffusion', 'rgcn')
                                for p in (ROOT/folder).glob('*.py')], ROOT)
    if local_source != '6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538':
        raise ValueError('Run this diagnostic on the unchanged restored baseline core')
    saved = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=False)
    if saved.get('version') != 2 or saved['provenance'] != config:
        raise ValueError('Checkpoint provenance mismatch')
    data = utils.load_data(args.dataset, data_root=args.data_root)
    snapshots = sum((utils.split_by_time(x) for x in (data.train, data.valid, data.test)), [])
    train, validation, _ = utils.spilt_data(snapshots)
    if saved['completed_stage'] != len(train)-1 or max(stages) >= len(train):
        raise ValueError('Use a completed run and stages inside the dataset')
    validate_history_cache(train, data.num_rels, data_dir/'history_snap_v2')
    answers = utils.load_all_answers_for_time_filter(validation, data.num_rels, data.num_nodes, False)
    args.num_nodes, args.num_rels = data.num_nodes, data.num_rels
    words, static_relations, static = 0, 0, None
    if args.add_static_graph:
        triples = np.asarray(_read_triplets_as_list(str(data_dir/'e-w-graph.txt'), {}, {}, load_time=False))
        static_relations, words = len(np.unique(triples[:, 1])), len(np.unique(triples[:, 2]))
        triples[:, 2] += data.num_nodes
        static = utils.build_sub_graph(data.num_nodes+words, static_relations, triples, use_cuda, args.gpu)
    model, generator = make_models(args, data.num_nodes, data.num_rels, words, static_relations, len(train))
    for filename, key, module in [('model.pt', 'model', model), ('diffusion.pt', 'diffusion', generator)]:
        weights = torch.load(run_dir/filename, map_location='cpu', weights_only=True)
        if weights.keys() != saved[key].keys() or any(not torch.equal(v, saved[key][k]) for k,v in weights.items()):
            raise ValueError(f'{filename} differs from latest.pt')
        if any(not torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
            raise ValueError(f'Nonfinite {filename}')
        module.load_state_dict(weights)
        module.to(device).requires_grad_(False)
    model.eval()
    generator.train()  # preserves Original generation mode, paired features reused below
    if options.check_only:
        print(f'Validated Original checkpoint: stage={saved["completed_stage"]}, '
              f'entities={data.num_nodes}, relations={data.num_rels}, snapshots={len(train)}. '
              'No inference or training executed.', flush=True)
        return
    torch.manual_seed(options.seed)
    if use_cuda:
        torch.cuda.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in stages:
        if len(validation[t]) == 0:
            continue
        # HCP retrieval uses all known forward query subjects, exactly as final
        # Original eval. Sampling and matching never inspect held-out objects.
        rng = random.Random(options.seed+1000003*t)
        times = list(range(t)) if t-1 < args.reply_batch else rng.sample(list(range(t)), args.reply_batch)
        facts = set()
        subjects = {str(s) for s in validation[t][:, 0]}
        for index in times:
            with (data_dir/'history_snap_v2'/f'snap_{index}.pkl').open('rb') as stream:
                cache = pickle.load(stream)
            for subject in subjects:
                facts.update(cache.get(subject, ()))
        tags = torch.as_tensor(list(facts), dtype=torch.long, device=device).reshape(-1, 3)
        selected = rng.sample(range(len(validation[t])), min(options.facts_per_stage, len(validation[t])))
        query = torch.as_tensor(validation[t][selected], dtype=torch.long, device=device)
        inverse = query[:, [2, 1, 0]].clone()
        inverse[:, 1] += data.num_rels
        query = torch.cat((query, inverse))
        history = [utils.build_sub_graph(data.num_nodes, data.num_rels, x, use_cuda, args.gpu)
                   for x in train[max(0, t-args.test_history_len):t]]
        # For t>=2, author's guidance receives the immediately previous eval
        # slice's graph window. Reconstruct it even when skipped in this subset.
        guide_history = [utils.build_sub_graph(data.num_nodes, data.num_rels, x, use_cuda, args.gpu)
                         for x in train[max(0, t-1-args.test_history_len):t-1]]
        with torch.no_grad():
            features = (generator(None, tags, args, False, use_cuda, static_graph=static,
                ct=t, model=model, history_glist=guide_history, triples=None)[1].detach()
                if len(tags) else None)
            if features is not None:
                entities, bank = model.select_entity(tags, features, use_cuda)
                with selector_override(model, entities, bank):
                    baseline = entity_scores(model, history, static, query, features, tags, use_cuda)
            else:
                entities, bank = None, None
                baseline = entity_scores(model, history, static, query, None, tags, use_cuda)
            no_replay = entity_scores(model, history, static, query, None, tags, use_cuda)
            memo = {}
            for index, triple in enumerate(query):
                subject, relation, target = triple.tolist()
                key = (subject, relation)
                if key not in memo:
                    info = (pool_for_query(tags, features, subject, relation, data.num_rels,
                        options.seed+1000003*t+9176*subject+97*relation) if features is not None else (0,0,None,None))
                    total, matched, matching_vector, control_vector = info
                    variant = {}
                    for name, vector in [('matched', matching_vector), ('control', control_vector)]:
                        if vector is None:
                            variant[name] = baseline
                        else:
                            with selector_override(model, entities, bank, subject, vector):
                                variant[name] = entity_scores(model, history, static, query, features, tags, use_cuda)
                    memo[key] = (total, matched, variant)
                total, matched, variant = memo[key]
                row = {'stage': t, 'subject': subject, 'relation': relation, 'target': target,
                    'direction': 'forward' if index < len(selected) else 'inverse',
                    'replay_facts': len(tags), 'entity_occurrences': total,
                    'matching_occurrences': matched, 'supported': int(matched>0),
                    'changes_pool': int(0<matched<total)}
                for name, score in [('original',baseline), ('matched',variant['matched']),
                                    ('control',variant['control']), ('no_replay',no_replay)]:
                    row[name+'_rank'] = filtered_rank(triple, score[index], answers[t])
                rows.append(row)
        # No graph, scores or feature cache is retained across stages.
        del memo, baseline, no_replay, features
        print(f'Stage {t}: queries={len(query)}, replay_facts={len(tags)}, '
              f'matched_queries={sum(r["supported"] for r in rows if r["stage"]==t)}', flush=True)
    if not rows:
        raise ValueError('No validation queries to diagnose')
    modes = ('original', 'matched', 'control', 'no_replay')
    supported = [r for r in rows if r['supported']]
    changing = [r for r in rows if r['changes_pool']]
    summary = {'protocol': 'paired-context-diagnostic-v1', 'training_provenance': config,
        'diagnostic_args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        'baseline_core_sha256': local_source, 'model_fusion_alpha': float(model.alpha.detach().cpu()),
        'queries': len(rows), 'supported_queries': len(supported), 'pool_changed_queries': len(changing),
        'overall': {mode:metrics(rows, mode+'_rank') for mode in modes},
        'supported': {mode:metrics(supported, mode+'_rank') for mode in modes},
        'pool_changed': {mode:metrics(changing, mode+'_rank') for mode in modes},
        'by_stage': {t:{mode:metrics([r for r in rows if r['stage']==t],mode+'_rank') for mode in modes} for t in stages},
        'limits': 'Final Original checkpoint, sampled VALIDATION queries, subject-row-only pooling intervention. Not paper metrics or chronological forgetting. No optimization. A null result does not rule out a trained relation-aware model.'}
    with (output/'queries.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('queries','supported_queries','pool_changed_queries','overall','pool_changed')},indent=2))
    print(f'Completed without training. Outputs: {output}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
