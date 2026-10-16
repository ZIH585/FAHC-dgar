"""Read-only, frozen mean+Shared checkpoint screen; VALIDATION only.

This is a post-hoc probe of a FINAL trained backbone, NOT an online training
result or the Original final evaluation protocol. Only a fresh small router is
updated using sampled TRAIN facts. No TEST query is scored or used for routing.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.cur import CURRuntime, isolated_predictor, paired_observations, utility_summary
from src.rrgcn import RecurrentRGCN
from src.runtime import UPSTREAM_COMMIT, fingerprint, validate_history_cache

PGR_SOURCE = '5e56dd4ae17f5ff08a0ad072f1bfeb2187f840f75adaf5d73bb0922bc63a481c'


def migrate_weights(model, weights):
    """Two adapter key renames only; the fresh router is the sole missing state."""
    mapping = {f'pgr_adapter.{factor}.weight': f'cur_adapter.{factor}.weight' for factor in ('up', 'down')}
    if not set(mapping) <= weights.keys():
        raise ValueError('Missing mean+Shared adapter weights')
    migrated = {mapping.get(key, key): value for key, value in weights.items()}
    expected = model.state_dict()
    if set(migrated) - expected.keys() or set(expected) - migrated.keys() != {key for key in expected if key.startswith('cur_router.')}:
        raise ValueError('Backbone checkpoint keys differ; only a fresh CUR router may be absent')
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
    args.cur_mode, args.cur_adapter_rank = 'utility', saved['pgr_adapter_rank']
    args.cur_probe_facts, args.cur_diagnostics = options.facts, False
    args.gpu, args.data_root, args.run_dir = options.gpu, options.data_root.resolve(), directory
    validate_args(args)
    torch.default_generator.manual_seed(args.seed)
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
    model.cur_router.requires_grad_(True)
    generator.to(device).requires_grad_(False).train()
    return args, config, train, validation, static, model, generator, device


def pick(rows, limit, seed, device):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=generator)[:limit].numpy()
    return torch.as_tensor(rows[indices], dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stages', default='60,120,180,240,300')
    parser.add_argument('--facts', type=int, default=128)
    parser.add_argument('--check-only', action='store_true', help='Validate/load only; no forward or router updates')
    options = parser.parse_args()
    selected = sorted(set(int(value) for value in options.stages.split(',')))
    if not selected or selected[0] < 2 or options.facts < 1:
        parser.error('need stages >=2 and a positive facts budget')
    output = options.output.resolve()
    if output == options.run_dir.resolve() or options.run_dir.resolve() in output.parents or output.exists():
        parser.error('output must be a new directory outside the source run')
    args, config, train, valid, static, model, generator, device = load(options)
    if selected[-1] >= len(train):
        parser.error('stage exceeds the source chronological sequence')
    if options.check_only:
        print('Validated frozen mean+Shared weights, dataset and cache. No forward/update performed.')
        return
    # Only selected stages fit the small router; no backbone or diffusion update.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = CURRuntime(args, model)
    stages, rows = [], []
    empty = torch.empty(0, 3, dtype=torch.long, device=device)
    for stage in range(1, selected[-1] + 1):
        if stage not in selected:
            runtime.fit(stage, model, [], empty, static, None, None)
            runtime.complete(stage)
            continue
        history = [utils.build_sub_graph(args.num_nodes, args.num_rels, snap, device.type == 'cuda', args.gpu)
                   for snap in train[max(0, stage - args.train_history_len):stage]]
        previous = [utils.build_sub_graph(args.num_nodes, args.num_rels, snap, device.type == 'cuda', args.gpu)
                    for snap in train[max(0, stage - 1 - args.train_history_len):stage - 1]]
        current = pick(train[stage], options.facts, args.seed + stage, device)
        validation = pick(valid[stage], options.facts, args.seed + stage + 1, device)
        def features(queries):
            with torch.no_grad():
                return generator.gereration_feature(args, stage, generator, model, previous,
                    None, static, queries.detach().cpu().numpy(), queries, device.type == 'cuda', train)[1:]
        with isolated_predictor(model):
            replay, tags = features(current)
            runtime.fit(stage, model, history, current, static, replay, tags)
            runtime.complete(stage)
            replay, tags = features(validation)
            if replay is None or not len(replay):
                stages.append({'stage': stage, 'reason': 'no_validation_replay'})
                continue
            triples = model.get_all_triple(validation)
            observed = paired_observations(model, history, static, triples, replay, tags)
            indices = torch.arange(len(triples), device=device)
            summary = utility_summary(observed, indices, observed['use_replay'])
            summary['direction_gain_vs_replay'] = utility_summary(observed, indices, triples[:, 1] >= args.num_rels)['gain_vs_replay']
            stages.append({'stage': stage, 'validation_raw_midrank': summary, 'train_probe': runtime.stats})
            passive = {key: observed[key].cpu().tolist() for key in ('rr_r', 'rr_0', 'probability', 'use_replay')}
            for index, (s, r, o) in enumerate(triples.cpu().tolist()):
                rows.append({'stage': stage, 'subject': s, 'relation': r, 'target': o,
                             **{key: value[index] for key, value in passive.items()}})
        print(json.dumps(stages[-1], allow_nan=False), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'kind': 'posthoc_frozen_checkpoint_probe_not_online_or_final_MRR', 'source': config,
               'stages': stages, 'test_queries_scored': 0, 'backbone_updates': 0}
    available = [s['validation_raw_midrank'] for s in stages if 'validation_raw_midrank' in s]
    if available:
        summary['macro_stage_probe'] = {key: sum(s[key] for s in available) / len(available)
            for key in ('gain_vs_replay', 'oracle_gain_vs_replay', 'direction_gain_vs_replay',
                        'weighted_sign_accuracy', 'replay_fraction')}
        summary['improved_stages'] = sum(s['gain_vs_replay'] > 0 for s in available)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if rows:
        with (output / 'queries.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'CUR frozen diagnostic: {output}')


if __name__ == '__main__':
    main()
