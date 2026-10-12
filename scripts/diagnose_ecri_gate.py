"""Fixed-checkpoint VALIDATION interventions; no training or checkpoint writes.

Replay is generated once per selected slice, before all fusion interventions.
scalar_alpha uses the ECRI-trained checkpoint, not a separately trained baseline.
"""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import pickle
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.ecri import EvolutionConfig, EvolutionHistory
from src.rrgcn import RecurrentRGCN
from src.runtime import fingerprint, isolated_rng, validate_history_cache


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--stages', default='90,180,300')
    parser.add_argument('--query-limit', type=int, default=16,
                        help='Maximum forward validation queries per selected slice')
    parser.add_argument('--shuffle-repeats', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='Check paths and source; no model/data loading')
    parser.add_argument('--allow-source-mismatch', action='store_true',
                        help='Explicit forensic override; results must report the mismatch')
    return parser


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    paths = [p for name in ('src', 'diffusion', 'rgcn') for p in (root/name).glob('*.py')]
    import hashlib
    return {'source_sha256': fingerprint(paths, root),
            'source_files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(paths)}}


def load_options(options):
    if options.query_limit < 1 or options.shuffle_repeats < 1 or options.gpu < -1:
        raise ValueError('Positive query-limit/shuffle-repeats and gpu >= -1 required')
    stages = sorted(set(int(t.strip()) for t in options.stages.split(',')))
    if not stages:
        raise ValueError('Select at least one validation stage')
    run = options.run_dir.resolve()
    for name in ('config.json', 'model.pt', 'diffusion.pt'):
        if not (run/name).is_file():
            raise FileNotFoundError(run/name)
    config = json.loads((run/'config.json').read_text())
    args = build_arg_parser().parse_args([])
    unknown = set(config['settings']) - set(vars(args))
    if unknown:
        raise ValueError(f'Unsupported saved settings: {sorted(unknown)}')
    vars(args).update(config['settings'])
    args.run_dir, args.data_root = run, options.data_root.resolve()
    args.gpu, args.test, args.resume = options.gpu, True, None
    validate_args(args)
    if not args.ecri:
        raise ValueError('This intervention requires an ECRI checkpoint')
    if min(stages) < 2*args.ecri_window:
        raise ValueError('Select stages after both complete ECRI windows')
    manifest = source_manifest()
    manifest['source_matches_config'] = manifest['source_sha256'] == config['source_sha256']
    manifest['saved_source_sha256'] = config['source_sha256']
    manifest['allow_source_mismatch'] = options.allow_source_mismatch
    # A dry run reports differences so the user can inspect the training checkout.
    if not manifest['source_matches_config'] and not (options.dry_run or options.allow_source_mismatch):
        raise ValueError('Source hash differs from training; run --dry-run on the training checkout first')
    return args, config, stages, manifest


def shuffled_features(features, kappa, seed):
    """Shuffle role/churn jointly within fixed support-count bins; keep c fixed."""
    result = features.copy()
    active = np.flatnonzero(features[:, 2] > 0)
    boundaries = np.array([n/(n+kappa) for n in (20, 40, 80)])
    bins = np.digitize(features[active, 2], boundaries)
    rng = np.random.default_rng(seed)
    moved = 0
    for bin_id in np.unique(bins):
        ids = active[bins == bin_id]
        source = rng.permutation(ids)
        result[ids, :2] = features[source, :2]
        moved += int(np.any(result[ids, :2] != features[ids, :2], axis=1).sum())
    return result, moved / len(active) if len(active) else 0.0


@contextmanager
def fusion_intervention(model, features, stage, mode):
    gate, alpha = model.ecri_gate, model.alpha.detach().clone()
    old_features, old_stage = model._ecri_features, model._ecri_stage
    try:
        model.set_ecri_context(features, stage)
        if mode in ('scalar_alpha', 'alpha025', 'alpha075', 'current_only'):
            model.ecri_gate = None
            if mode != 'scalar_alpha':
                value = {'alpha025': .25, 'alpha075': .75, 'current_only': 1.0}[mode]
                with torch.no_grad():
                    model.alpha.fill_(value)
        yield
    finally:
        model.ecri_gate = gate
        with torch.no_grad():
            model.alpha.copy_(alpha)
        model._ecri_features, model._ecri_stage = old_features, old_stage


def score_cases(model, graphs, static_graph, queries, replay, tags, features,
                stage, args, answers, shuffle_repeats):
    cases = [('full', features), ('bias_only', features.copy()),
             ('scalar_alpha', features), ('alpha025', features),
             ('alpha075', features), ('current_only', features)]
    cases[1][1][:, :2] = 0
    shuffle_moved = []
    for repeat in range(shuffle_repeats):
        shuffled, moved = shuffled_features(features, args.ecri_kappa,
                                            args.seed + 9000011*stage + repeat)
        cases.append((f'shuffle_{repeat}', shuffled)); shuffle_moved.append(moved)
    ranks = {}
    all_queries = None
    for mode, context in cases:
        with fusion_intervention(model, context, stage, mode), torch.no_grad():
            triples, scores, _ = model.predict(graphs, args.num_rels, static_graph, queries,
                replay, replay[:, -1] if replay is not None else None, tags, args.gpu >= 0)
            if not torch.isfinite(scores).all():
                raise ValueError(f'Nonfinite validation logits at stage {stage}, mode {mode}')
            _, _, _, rank = utils.get_total_rank(triples, scores, answers, 1000, rel_predict=0)
            ranks[mode] = rank.detach().cpu().numpy()
            if all_queries is None:
                all_queries = triples.detach().cpu().numpy()
    return all_queries, ranks, shuffle_moved


def create_models(args, data, count, device):
    static = None
    num_words = num_static_rels = 0
    if args.add_static_graph:
        triples = np.array(_read_triplets_as_list(str(args.data_root/args.dataset/'e-w-graph.txt'),
                                                {}, {}, load_time=False))
        num_static_rels = len(np.unique(triples[:, 1]))
        num_words = len(np.unique(triples[:, 2]))
        triples[:, 2] += data.num_nodes
        static = utils.build_sub_graph(data.num_nodes+num_words, num_static_rels,
                                       triples, device.type == 'cuda', args.gpu)
    model = RecurrentRGCN(args.decoder, args.encoder, data.num_nodes, data.num_rels,
        num_static_rels, num_words, args.n_hidden, args.attn_heads, args.opn,
        sequence_len=args.train_history_len, num_bases=args.n_bases, num_basis=args.n_basis,
        num_hidden_layers=args.n_layers, dropout=args.dropout, self_loop=args.self_loop,
        skip_connect=args.skip_connect, layer_norm=args.layer_norm,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout, aggregation=args.aggregation, weight=args.weight,
        discount=args.discount, angle=args.angle, use_static=args.add_static_graph,
        entity_prediction=args.entity_prediction, relation_prediction=args.relation_prediction,
        use_cuda=device.type == 'cuda', gpu=args.gpu, args=args).to(device)
    diffusion = Att_Diffuse_model(create_model_diffu(args), args, args.encoder,
        data.num_nodes, data.num_rels, dropout=args.dropout, max_time=count,
        num_words=num_words, num_static_rels=num_static_rels, num_bases=args.n_bases).to(device)
    model.load_state_dict(torch.load(args.run_dir/'model.pt', map_location='cpu', weights_only=True))
    diffusion.load_state_dict(torch.load(args.run_dir/'diffusion.pt', map_location='cpu', weights_only=True))
    for component in (model, diffusion):
        component.eval()
        component.requires_grad_(False)
    return model, diffusion, static


def metrics(ranks):
    return {'queries': len(ranks), 'mrr': float(np.mean(1/ranks)),
            **{f'hits@{k}': float(np.mean(ranks <= k)) for k in (1, 3, 10)}}


def run(options):
    args, config, stages, provenance = load_options(options)
    print(json.dumps({'stages': stages, 'forward_query_limit': options.query_limit,
                      **provenance}, indent=2), flush=True)
    if options.dry_run:
        print('Dry run only: checkpoints exist; no weights/data loaded, no CUDA initialized.')
        return
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise ValueError('Use a new output-dir to preserve earlier diagnostics')
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device unavailable; choose --gpu -1 explicitly for CPU')
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    data_path = args.data_root/args.dataset
    paths = [data_path/f'{n}.txt' for n in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        paths.append(data_path/'e-w-graph.txt')
    for key, actual in (
        ('data_sha256', fingerprint(paths, data_path)),
        ('history_sha256', fingerprint(list((data_path/'history_snap_v2').glob('snap_*.pkl')), data_path))):
        if actual != config[key]:
            raise ValueError(f'{key} differs from training')
        provenance[key] = actual
    data = utils.load_data(args.dataset, data_root=args.data_root)
    snapshots = []
    for split in (data.train, data.valid, data.test):
        snapshots.extend(utils.split_by_time(split))
    train, valid, _ = utils.spilt_data(snapshots)  # Original's per-slice protocol.
    if max(stages) >= len(train):
        raise ValueError('Selected stage outside the dataset')
    validate_history_cache(train, data.num_rels, data_path/'history_snap_v2')
    args.num_nodes, args.num_rels = data.num_nodes, data.num_rels
    torch.default_generator.manual_seed(args.seed)
    model, diffusion, static = create_models(args, data, len(train), device)
    alpha = float(model.alpha.detach().cpu())
    gate_values = {name: float(value.detach().cpu()) for name, value in model.ecri_gate.state_dict().items()}
    gate_report = {'bias': gate_values['bias'],
        'role_weight': float(torch.nn.functional.softplus(model.ecri_gate.role_raw.detach()).cpu()),
        'churn_weight': float(torch.nn.functional.softplus(model.ecri_gate.churn_raw.detach()).cpu())}
    stage_log = args.run_dir/'ecri.jsonl'
    if stage_log.is_file():
        last = json.loads(stage_log.read_text().splitlines()[-1])
        for key, value in gate_report.items():
            if abs(value-last[f'gate_{key}']) > 1e-6:
                raise ValueError('Checkpoint gate differs from the last stage log; use a completed matching run')
    provenance['checkpoint_gate_matches_stage_log'] = stage_log.is_file()
    print(json.dumps({'checkpoint_alpha': alpha, 'checkpoint_gate': gate_report}), flush=True)
    answers = utils.load_all_answers_for_time_filter(valid, data.num_rels, data.num_nodes, False)
    evolution = EvolutionHistory(data.num_nodes, data.num_rels, EvolutionConfig.from_args(args))
    output, stage_records = [], []
    for stage in range(max(stages)+1):
        if stage in stages:
            features = evolution.features(stage)
            selected = np.random.default_rng(args.seed+6000011*stage).permutation(len(valid[stage]))[:options.query_limit]
            if not len(selected):
                raise ValueError(f'No validation queries at stage {stage}')
            queries = torch.as_tensor(valid[stage][selected], dtype=torch.long, device=device)
            graphs = [utils.build_sub_graph(data.num_nodes, data.num_rels, triples,
                      device.type == 'cuda', args.gpu) for triples in train[max(0,stage-args.test_history_len):stage]]
            with isolated_rng(args.seed+7000003+stage, device), torch.no_grad():
                indices = list(range(stage)) if stage-1 < args.reply_batch else random.sample(range(stage), args.reply_batch)
                facts = set()
                for t in indices:
                    with (data_path/'history_snap_v2'/f'snap_{t}.pkl').open('rb') as stream:
                        history = pickle.load(stream)
                    for s in valid[stage][selected, 0]:
                        facts.update(history.get(str(s), ()))
                tags = torch.as_tensor(sorted(facts), dtype=torch.long, device=device).reshape(-1, 3)
                replay = None
                if len(tags):
                    replay = diffusion(None, tags, args, False, device.type == 'cuda',
                        static_graph=static, ct=stage, model=model, history_glist=graphs, triples=None)[1].detach()
            triples, ranks, moved = score_cases(model, graphs, static, queries, replay, tags,
                features, stage, args, answers[stage], options.shuffle_repeats)
            weights = model.ecri_gate
            beta = float(torch.nn.functional.softplus(weights.role_raw.detach()).cpu())
            gamma = float(torch.nn.functional.softplus(weights.churn_raw.detach()).cpu())
            hcp_entities = facts_entities(tags)
            for i,(s,r,o) in enumerate(triples):
                row = {'snapshot': stage, 'forward_row': int(selected[i % len(selected)]),
                       'inverse': int(i >= len(selected)), 'subject': int(s), 'relation': int(r),
                       'target': int(o), 'confidence': float(features[s,2]),
                       'evolution_logit': float(beta*features[s,0]+gamma*features[s,1]),
                       'subject_in_hcp': int(s) in hcp_entities,
                       **{f'rank_{mode}': int(values[i]) for mode,values in ranks.items()}}
                output.append(row)
            stage_records.append({'snapshot': stage, 'hcp_facts': len(tags), 'sampled_forward_rows': selected.tolist(),
                                  'shuffle_changed_fraction': moved,
                                  'metrics': {mode: metrics(values) for mode,values in ranks.items()}})
            print(f'Stage {stage}: {len(selected)} forward validation queries, {len(tags)} fixed HCP facts', flush=True)
        evolution.observe(stage, train[stage])
    all_ranks = {mode: np.array([row[f'rank_{mode}'] for row in output]) for mode in ranks}
    reliable = [row for row in output if row['confidence']>0]
    median = float(np.median([row['evolution_logit'] for row in reliable])) if reliable else None
    groups = {}
    for name,rows in (('low', [r for r in reliable if r['evolution_logit']<=median]),
                      ('high', [r for r in reliable if r['evolution_logit']>median])):
        if not rows:
            groups[name] = {'queries': 0}; continue
        groups[name] = {'queries': len(rows), 'median_threshold': median,
            'mean_rr_delta_alpha075_minus_alpha025': float(np.mean([1/r['rank_alpha075']-1/r['rank_alpha025'] for r in rows])),
            'mean_rr_delta_full_minus_bias_only': float(np.mean([1/r['rank_full']-1/r['rank_bias_only'] for r in rows]))}
    summary = {'split': 'VALIDATION', 'checkpoint': str(args.run_dir/'model.pt'),
        'checkpoint_alpha': alpha, 'checkpoint_gate_raw': gate_values,
        'same_checkpoint_for_all_modes': True, 'replay_generated_once_per_stage': True,
        'all_stages_have_nonempty_hcp': all(r['hcp_facts']>0 for r in stage_records),
        'not_original_trained_baseline': 'scalar_alpha retains all ECRI-trained parameters',
        'scope': 'Sampled final-model validation interventions; no training or test evaluation',
        'query_subject_grouping_limitation': 'Candidate embedding changes can also affect ranks; groups are not causal attribution',
        'inverse_queries_are_paired': True, 'provenance': provenance,
        'stage_records': stage_records, 'pooled_metrics': {mode: metrics(values) for mode,values in all_ranks.items()},
        'supported_query_subject_groups': groups,
        'all_parameter_grads_none': all(p.grad is None for m in (model,diffusion) for p in m.parameters())}
    options.output_dir.mkdir(parents=True,exist_ok=True)
    with (options.output_dir/'queries.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]),lineterminator='\n');writer.writeheader();writer.writerows(output)
    (options.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['pooled_metrics'],indent=2))
    print(f'Output: {options.output_dir.resolve()}; checkpoint files were read only.')


def facts_entities(tags):
    return set(tags[:,(0,2)].detach().cpu().reshape(-1).tolist())


if __name__ == '__main__':
    run(build_parser().parse_args())
