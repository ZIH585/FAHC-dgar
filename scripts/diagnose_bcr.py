"""Paired VALIDATION-only BCR screen on an existing Original checkpoint; no training."""
import argparse
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
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import PROTOCOL, UPSTREAM_COMMIT, fingerprint, validate_history_cache
from src.bcr import BCR, isolated_generation, predict


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed Original checkpoint, read-only')
    p.add_argument('--output', type=Path, required=True, help='Fresh directory outside the checkpoint run')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--stages', default='60,120,180,240,300')
    p.add_argument('--seed', type=int, default=1, help='Diagnostic sampling/noise seed, not a new model seed')
    p.add_argument('--check-only', action='store_true', help='Validate/load only; stop before inference')
    return p


def metrics(rows, column):
    if not rows:
        return None
    ranks = np.asarray([r[column] for r in rows], dtype=float)
    return dict(queries=len(rows), mrr=float((1/ranks).mean()),
                hits1=float((ranks<=1).mean()), hits3=float((ranks<=3).mean()), hits10=float((ranks<=10).mean()))


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


def load_original(options):
    run_dir = options.run_dir.resolve()
    config = json.loads((run_dir / 'config.json').read_text())
    s = config['settings']
    local_hash = fingerprint([p for folder in ('src', 'diffusion', 'rgcn')
                              for p in (ROOT / folder).glob('*.py')], ROOT)
    sources = {'6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538',
               'c22dc00022ff00ef8ebf9486d32f839ef9f60b5e0c59d04b79413401ba96b3d5', local_hash}
    if (config['protocol'] != PROTOCOL or config['upstream_commit'] != UPSTREAM_COMMIT
            or config['source_sha256'] not in sources
            or s.get('grb_mode', 'original') != 'original'
            or s.get('bcr_mode', 'original') != 'original'
            or any(k.startswith(('fahc', 'ecri', 'qcdr')) for k in s)
            or not s['diffuc'] or s['delete_feature_reply'] or s['delete_his_prompt']):
        raise ValueError('Use a completed, restored Original DGAR checkpoint')
    args = build_arg_parser().parse_args([])
    for k, v in s.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.data_root, args.gpu = options.data_root.resolve(), options.gpu
    args = validate_args(args)
    cuda = options.gpu >= 0
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; --gpu -1 explicitly selects CPU')
    device = torch.device(f'cuda:{options.gpu}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    data_dir = args.data_root / args.dataset
    files = [data_dir / f'{x}.txt' for x in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        files.append(data_dir / 'e-w-graph.txt')
    if (fingerprint(files, data_dir) != config['data_sha256']
            or fingerprint(list((data_dir / 'history_snap_v2').glob('snap_*.pkl')), data_dir)
            != config['history_sha256']):
        raise ValueError('Data/cache do not match the checkpoint')
    saved = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=False)
    if saved.get('version') != 2 or saved['provenance'] != config:
        raise ValueError('Checkpoint provenance mismatch')
    data = utils.load_data(args.dataset, data_root=args.data_root)
    snapshots = sum((utils.split_by_time(x) for x in (data.train, data.valid, data.test)), [])
    train, validation, _ = utils.spilt_data(snapshots)
    if saved['completed_stage'] != len(train) - 1:
        raise ValueError('Checkpoint training is incomplete')
    validate_history_cache(train, data.num_rels, data_dir / 'history_snap_v2')
    args.num_nodes, args.num_rels = data.num_nodes, data.num_rels
    words, static_relations, static = 0, 0, None
    if args.add_static_graph:
        triples = np.asarray(_read_triplets_as_list(str(data_dir / 'e-w-graph.txt'), {}, {}, load_time=False))
        static_relations, words = len(np.unique(triples[:, 1])), len(np.unique(triples[:, 2]))
        triples[:, 2] += data.num_nodes
        static = utils.build_sub_graph(data.num_nodes + words, static_relations, triples, cuda, args.gpu)
    model, generator = make_models(args, data.num_nodes, data.num_rels, words, static_relations, len(train))
    for filename, key, module in [('model.pt', 'model', model), ('diffusion.pt', 'diffusion', generator)]:
        weights = torch.load(run_dir / filename, map_location='cpu', weights_only=True)
        if weights.keys() != saved[key].keys() or any(not torch.equal(v, saved[key][k]) for k, v in weights.items()):
            raise ValueError(f'{filename} differs from latest.pt')
        if any(not torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
            raise ValueError(f'Nonfinite {filename}')
        module.load_state_dict(weights)
        module.to(device).requires_grad_(False)
    model.eval()
    generator.train()
    return args, config, local_hash, train, validation, static, model, generator, device, cuda


def run(options):
    if options.gpu < -1 or options.seed < 0:
        raise ValueError('Invalid GPU/seed')
    stages = [int(x) for x in options.stages.split(',')]
    if not stages or stages != sorted(set(stages)) or min(stages) < 2:
        raise ValueError('Stages must be unique, increasing and >=2')
    output, run_dir = options.output.resolve(), options.run_dir.resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError('Keep outputs outside the checkpoint directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    args, config, source_hash, train, validation, static, model, generator, device, cuda = load_original(options)
    if max(stages) >= len(train):
        raise ValueError('Stage outside dataset')
    if options.check_only:
        print(f'Validated Original: stages={len(train)}, entities={args.num_nodes}, '
              f'relations={args.num_rels}. Stopped before inference/training.', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    # Private config: never change the input run's config or call prepare_run.
    args.run_dir = output
    args.bcr_mode = 'incident'
    manager = BCR(args, train, args.num_rels, args.num_nodes)
    answers = utils.load_all_answers_for_time_filter(validation, args.num_rels, args.num_nodes, False)
    rows, costs, counts = [], [], []
    modes = ('original', 'outgoing', 'incident', 'random')
    for t in stages:
        if not len(validation[t]):
            continue
        rng = random.Random(options.seed + 1000003 * t)
        times = list(range(t)) if t - 1 < args.reply_batch else rng.sample(list(range(t)), args.reply_batch)
        noise_seed = rng.getrandbits(63)
        triples = torch.as_tensor(validation[t], dtype=torch.long, device=device)
        history = [utils.build_sub_graph(args.num_nodes, args.num_rels, x, cuda, args.gpu)
                   for x in train[max(0, t - args.test_history_len):t]]
        # This is the same retrospective guidance window as the author at t>=2.
        guide = [utils.build_sub_graph(args.num_nodes, args.num_rels, x, cuda, args.gpu)
                 for x in train[max(0, t - 1 - args.test_history_len):t - 1]]
        legacy = set()
        for index in times:
            with (manager.cache_dir / f'snap_{index}.pkl').open('rb') as stream:
                cache = pickle.load(stream)
            for subject in set(map(int, validation[t][:, 0])):
                legacy.update(tuple(map(int, x)) for x in cache.get(str(subject), ()))
        ranks = {}
        for mode in modes:
            start = time.monotonic()
            with torch.no_grad():
                if mode == 'original':
                    with isolated_generation(generator, model, device, noise_seed):
                        replay = manager.generate(args, list(legacy), generator, model,
                            guide, None, static, t, device, cuda)
                    expanded, scores, _ = model.predict(history, args.num_rels, static,
                        triples, replay.features, replay.objects, replay.tags, cuda)
                    numbers = [len(replay.tags)]
                else:
                    args.bcr_mode = mode
                    replays = manager.evaluation(t, validation[t][:, 0].tolist(),
                        validation[t][:, 2].tolist(), generator, model, guide, None,
                        static, device, cuda, final=True, times=times, stop=t,
                        seed=noise_seed, phase=f'diagnostic-{mode}')
                    expanded, scores, _ = predict(model, history, args.num_rels,
                        static, triples, replays, cuda)
                    numbers = [len(x.tags) for x in replays]
                ranks[mode] = utils.get_total_rank(expanded, scores.clone(), answers[t], eval_bz=1000)[3].cpu().tolist()
                del scores
            costs.append(dict(stage=t, mode=mode, seconds=time.monotonic() - start))
            counts.append(dict(stage=t, mode=mode, generated_facts=numbers, times=times))
        expanded = expanded.cpu().tolist()
        for i, (s, r, o) in enumerate(expanded):
            rows.append(dict(stage=t, subject=s, relation=r, target=o,
                direction='forward' if i < len(triples) else 'inverse',
                **{mode + '_rank': int(ranks[mode][i]) for mode in modes}))
        print(f'Stage {t}: {len(expanded)} validation queries; '
              + ', '.join(f'{m} MRR={metrics([r for r in rows if r["stage"]==t], m+"_rank")["mrr"]:.6f}' for m in modes), flush=True)
    if not rows:
        raise ValueError('No validation queries')
    by_stage = {str(t): {m: metrics([r for r in rows if r['stage'] == t], m + '_rank') for m in modes}
                for t in stages if any(r['stage'] == t for r in rows)}
    overall = {m: metrics(rows, m + '_rank') for m in modes}
    delta = {m: overall['incident']['mrr'] - overall[m]['mrr'] for m in modes if m != 'incident'}
    nonnegative = sum(all(x['incident']['mrr'] >= x[m]['mrr'] for m in ('original', 'outgoing', 'random'))
                      for x in by_stage.values())
    decisions = dict(delta_mrr=delta, stages_not_worse_than_all_controls=nonnegative,
        provisional_gate=all(x >= .003 for x in delta.values()) and nonnegative >= 4,
        note='Resource gate, not statistical significance. Inspect query concentration before training.')
    summary = dict(protocol='paired-bcr-validation-v1', training_provenance=config,
        diagnostic_source_sha256=source_hash,
        diagnostic_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
        overall=overall, by_stage=by_stage,
        by_direction={d: {m: metrics([r for r in rows if r['direction'] == d], m + '_rank') for m in modes}
                      for d in ('forward', 'inverse')}, counts=counts, costs=costs, decision=decisions,
        limits='Final trained Original checkpoint, selected slices FULL VALIDATION; paired isolated generation, not original full evaluation RNG trajectory or paper Average. One sampled HCP/noise realization. No optimization. Training utility remains untested.')
    with (output / 'queries.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(overall=overall, decision=decisions), indent=2), flush=True)
    print(f'Completed without training. Outputs: {output}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
