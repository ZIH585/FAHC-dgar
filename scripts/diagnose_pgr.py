"""Frozen Shared checkpoint diagnostic, VALIDATION only, no parameter updates.

Use one guided generation per stage. Mapping controls reuse its exact features;
noise trials change only the terminal Gaussian draw, never denoiser/guidance.
This isolated partial-slice diagnostic is NOT DGAR final Average evaluation.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.pgr import PGRRuntime, aggregate, bank_variance, isolated_predictor
from src.pgr_diagnostics import metrics
from src.rrgcn import RecurrentRGCN
from src.runtime import UPSTREAM_COMMIT, fingerprint, rng_state, restore_rng, validate_history_cache

SHARED_SOURCE = '8211d3e7c04660089b4d3537e6e7ad51bebf01e6836da90c164c44f711772168'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed audited RAHA Shared checkpoint, read-only')
    p.add_argument('--output', type=Path, required=True, help='New directory, outside input run')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--stages', default='60,120,180,240,300')
    p.add_argument('--noise-trials', type=int, default=4)
    p.add_argument('--seed', type=int, default=1, help='Diagnostic seed, not a new training seed')
    p.add_argument('--check-only', action='store_true', help='Validate/load on CPU or GPU, without any forward')
    return p


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


def load_shared(options):
    """Strict read-only migration of TWO adapter key names, no other remapping."""
    directory = options.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    saved_args = config['settings']
    if (config['protocol'] != 'raha-shared-v1' or config['upstream_commit'] != UPSTREAM_COMMIT
            or config['source_sha256'] != SHARED_SOURCE or saved_args.get('raha_mode') != 'shared'
            or not saved_args['diffuc'] or saved_args['delete_feature_reply']
            or saved_args['delete_his_prompt'] or saved_args['delete_score'] or saved_args['vl']):
        raise ValueError('Expected the audited, completed Shared checkpoint; unknown methods are rejected')
    args = build_arg_parser().parse_args([])
    for key, value in saved_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.pgr_mode, args.pgr_adapter_rank = 'shared', saved_args['raha_rank']
    args.gpu, args.data_root, args.run_dir = options.gpu, options.data_root.resolve(), directory
    args = validate_args(args)
    args._pgr_runtime = PGRRuntime(args)
    cuda = args.gpu >= 0
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --gpu -1 explicitly for CPU')
    device = torch.device(f'cuda:{args.gpu}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    data_dir = args.data_root / args.dataset
    files = [data_dir / f'{x}.txt' for x in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        files.append(data_dir / 'e-w-graph.txt')
    if (fingerprint(files, data_dir) != config['data_sha256']
            or fingerprint(list((data_dir / 'history_snap_v2').glob('snap_*.pkl')), data_dir)
            != config['history_sha256']):
        raise ValueError('Data/history cache differ from the checkpoint')
    latest = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=False)
    if latest.get('version') != 2 or latest['provenance'] != config:
        raise ValueError('Checkpoint provenance/version mismatch')
    data = utils.load_data(args.dataset, data_root=args.data_root)
    snapshots = sum((utils.split_by_time(x) for x in (data.train, data.valid, data.test)), [])
    train, validation, _ = utils.spilt_data(snapshots)
    if latest['completed_stage'] != len(train) - 1:
        raise ValueError('Checkpoint is not a completed chronological run')
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
        weights = torch.load(directory / filename, map_location='cpu', weights_only=True)
        if weights.keys() != latest[key].keys() or any(not torch.equal(v, latest[key][k]) for k, v in weights.items()):
            raise ValueError(f'{filename} differs from latest.pt')
        if any(not torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
            raise ValueError(f'Nonfinite {filename}')
        if key == 'model':
            mapping = {f'raha.shared.{factor}.weight': f'pgr_adapter.{factor}.weight'
                       for factor in ('up', 'down')}
            if not set(mapping) <= weights.keys():
                raise ValueError('Missing Shared adapter weights')
            weights = {mapping.get(k, k): v for k, v in weights.items()}
        module.load_state_dict(weights, strict=True)
        module.to(device).requires_grad_(False)
    model.eval()
    generator.train()  # Original generator dropout/BN; restored between stages.
    return args, config, train, validation, static, model, generator, device, cuda


@contextmanager
def stage_state(model, generator, device, seed):
    state = rng_state(device)
    buffers = [(b, b.detach().clone()) for b in generator.buffers()]
    modes = [(m, m.training) for m in generator.modules()]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    try:
        with isolated_predictor(model):
            yield
    finally:
        with torch.no_grad():
            for b, value in buffers:
                b.copy_(value)
        for module, training in modes:
            module.training = training
        restore_rng(state, device)


@torch.no_grad()
def capture_once(generator, model, args, tags, graphs, static, stage, cuda):
    sampler, captured = generator.diffu, {}
    original = sampler.p_mean_variance
    had_override = 'p_mean_variance' in sampler.__dict__
    old_override = sampler.__dict__.get('p_mean_variance')

    def capture(*a, **kw):
        if captured:
            raise ValueError('Expected exactly one guided generation per stage')
        mean, log_variance, x0 = original(*a, **kw)
        sigma = (a[4] != 0).to(mean).view(-1, 1) * (.5 * log_variance).exp()[:, :1]
        captured.update(mean=mean.detach(), sigma=sigma.detach(), x0=x0.detach())
        return mean, log_variance, x0

    sampler.p_mean_variance = capture
    try:
        features = generator(None, tags, args, False, cuda, static_graph=static,
            ct=stage, model=model, history_glist=graphs, triples=None)[1].detach()
    finally:
        if had_override:
            sampler.p_mean_variance = old_override
        else:
            del sampler.p_mean_variance
    if not captured:
        raise ValueError('Sampler capture was not reached')
    return features, captured


def norm_match(bank, reference):
    length = bank.norm(dim=-1, keepdim=True)
    target = reference.norm(dim=-1, keepdim=True)
    if ((length == 0) & (target > 1e-8)).any():
        raise ValueError('Cannot match a nonzero reference using a zero direction')
    return bank * target / length.clamp_min(1e-20)


@torch.no_grad()
def mapping_controls(model, tags, features, mean_features, seed):
    ids, raw = aggregate(tags, features, model.num_ents)
    learned = model.pgr_adapter(raw)
    width = raw.shape[1]
    matrix = torch.eye(width, device=raw.device) + model.pgr_adapter.up.weight @ model.pgr_adapter.down.weight
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.linalg.qr(torch.randn(width, width, generator=generator))[0].to(raw)
    randomized = rotation @ matrix @ rotation.T  # Singular spectrum of A preserved.
    scaled_features = features.clone()
    gains = []
    for position in (0, 2):
        values = features[:, position]
        gain = model.pgr_adapter(values).norm(dim=-1).mean() / values.norm(dim=-1).mean().clamp_min(1e-20)
        scaled_features[:, position] *= gain
        gains.append(float(gain.cpu()))
    _, scaled = aggregate(tags, scaled_features, model.num_ents)
    _, raw_mean = aggregate(tags, mean_features, model.num_ents)
    return ids, {'learned': learned, 'identity_norm': norm_match(raw, learned),
                 'random_spectrum_norm': norm_match(raw @ randomized.T, learned),
                 'fixed_role_gain': scaled, 'mean_shared': model.pgr_adapter(raw_mean)}, gains


@torch.no_grad()
def score_bank(model, ids, bank, tags, features, graphs, static, queries, cuda):
    had = 'select_entity' in model.__dict__
    previous = model.__dict__.get('select_entity')
    model.select_entity = lambda *a, **kw: (ids, bank)
    try:
        triples, logits, _ = model.predict(graphs, model.num_rels, static, queries,
                                           features, features[:, 2], tags, cuda)
        return triples, logits.detach()
    finally:
        if had:
            model.select_entity = previous
        else:
            del model.select_entity


def run(options):
    if options.seed < 0 or options.gpu < -1 or options.noise_trials < 2:
        raise ValueError('Invalid diagnostic seed, GPU or trial count')
    args, config, train, validation, static, model, generator, device, cuda = load_shared(options)
    stages = sorted(set(int(v) for v in options.stages.split(',')))
    if not stages or min(stages) < 2 or max(stages) >= len(train):
        raise ValueError('Diagnostic stages must be in [2, T-1]')
    if options.check_only:
        print(f'Validated Shared checkpoint, data/cache, finite weights and strict key migration; no forward. stages={stages}')
        return
    output = options.output.resolve()
    if output == options.run_dir.resolve() or options.run_dir.resolve() in output.parents:
        raise ValueError('Diagnostic output must be outside the input checkpoint run')
    output.mkdir(parents=True, exist_ok=False)
    answers = utils.load_all_answers_for_time_filter(validation, args.num_rels, args.num_nodes, False)
    rows, noise_rows, generations = [], [], []
    for stage in stages:
        with stage_state(model, generator, device, options.seed + stage * 104729):
            # Diagnostic uses each stage's own past training window for guidance.
            graphs = [utils.build_sub_graph(args.num_nodes, args.num_rels, snap, cuda, args.gpu)
                      for snap in train[max(0, stage - args.test_history_len):stage]]
            times = list(range(stage)) if stage - 1 < args.reply_batch else random.sample(list(range(stage)), args.reply_batch)
            historical = set()
            for time in times:
                with (args.data_root / args.dataset / 'history_snap_v2' / f'snap_{time}.pkl').open('rb') as stream:
                    index = pickle.load(stream)
                for query in validation[stage]:
                    historical.update(index.get(str(query[0]), ()))
            tags = torch.as_tensor(list(historical), dtype=torch.long, device=device).reshape(-1, 3)
            if not len(tags):
                raise ValueError(f'No historical facts at stage {stage}')
            features, capture = capture_once(generator, model, args, tags, graphs, static, stage, cuda)
            mean_features = torch.cat((capture['x0'][:, :2], capture['mean'][:, None]), dim=1)
            ids, controls, gains = mapping_controls(model, tags, features, mean_features, options.seed + stage)
            queries = torch.as_tensor(validation[stage], dtype=torch.long, device=device)
            base_logits, expanded = None, None
            for arm, bank in controls.items():
                triples, logits = score_bank(model, ids, bank, tags, features, graphs, static, queries, cuda)
                if arm == 'mean_shared':
                    base_logits, expanded = logits, triples
                _, _, raw_rank, filtered_rank = utils.get_total_rank(triples, logits.clone(), answers[stage], eval_bz=1000)
                for (s, r, o), raw, filtered in zip(triples.cpu().tolist(), raw_rank.cpu().tolist(), filtered_rank.cpu().tolist()):
                    rows.append(dict(stage=stage, arm=arm, subject=s, relation=r, target=o,
                                     inverse=int(r >= args.num_rels), rank_raw=int(raw), rank_filtered=int(filtered)))
            # Fix competitors using mean scores, without consulting query targets.
            competitors = base_logits.topk(min(2, args.num_nodes), dim=1).indices
            margins, ranks, tops = [], [], []
            observed_banks = []
            private = torch.Generator(device=device).manual_seed(options.seed + stage * 1009)
            for trial in range(options.noise_trials):
                noise = torch.randn(capture['mean'].shape, generator=private, device=device)
                trial_features = torch.cat((capture['x0'][:, :2],
                    (capture['mean'] + capture['sigma'] * noise)[:, None]), dim=1)
                _, raw_bank = aggregate(tags, trial_features, args.num_nodes)
                observed_banks.append(raw_bank)
                _, logits = score_bank(model, ids, model.pgr_adapter(raw_bank), tags,
                    trial_features, graphs, static, queries, cuda)
                selected = logits.gather(1, competitors)
                margins.append(selected[:, 0] - selected[:, 1])
                tops.append(logits.argmax(dim=1))
                _, _, _, rank = utils.get_total_rank(expanded, logits.clone(), answers[stage], eval_bz=1000)
                ranks.append(rank)
            margin_var = torch.stack(margins).var(dim=0, unbiased=True)
            rank_stack, top_stack = torch.stack(ranks), torch.stack(tops)
            base_top = competitors[:, 0]
            flip = (top_stack != base_top).float().mean(dim=0)
            empirical = torch.stack(observed_banks).var(dim=0, unbiased=True).mean(dim=1)
            analytic = bank_variance(tags, capture['sigma'], args.num_nodes)
            valid = analytic > 0
            ratio = float((empirical[valid] / analytic[valid]).mean().cpu()) if valid.any() else None
            for i, (s, r, o) in enumerate(expanded.cpu().tolist()):
                noise_rows.append(dict(stage=stage, subject=s, relation=r, target=o,
                    inverse=int(r >= args.num_rels), top2_margin_variance=float(margin_var[i].cpu()),
                    top1_flip_fraction=float(flip[i].cpu()), rank_min=int(rank_stack[:, i].min().cpu()),
                    rank_max=int(rank_stack[:, i].max().cpu())))
            generations.append(dict(stage=stage, times=times, facts=len(tags), denoiser_calls=1,
                noise_trials=options.noise_trials, role_gains=gains,
                empirical_over_analytic_variance_mean=ratio,
                warning='Few noise trials give a coarse conditional variance estimate, not a performance test'))
            print(f'Stage {stage}: facts={len(tags)}; controls={len(controls)}; noise trials={options.noise_trials}')
    for name, values in [('queries.csv', rows), ('noise_queries.csv', noise_rows)]:
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / 'generation.jsonl').write_text(''.join(json.dumps(g, allow_nan=False) + '\n' for g in generations))
    summary = dict(protocol='pgr-frozen-shared-diagnostic-v1', checkpoint=config,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=fingerprint([p for folder in ('src', 'diffusion', 'rgcn') for p in (ROOT / folder).glob('*.py')], ROOT),
        stages=stages, diagnostic_seed=options.seed, noise_trials=options.noise_trials,
        note='Partial VALIDATION slices; own past-train guidance; isolated RNG; NOT paper Average/full TEST',
        arms={arm: {group: metrics([r for r in rows if r['arm'] == arm and (group == 'all' or r['inverse'] == int(group == 'inverse'))])
                    for group in ('all', 'forward', 'inverse')} for arm in controls})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Diagnostic output: {output}')


if __name__ == '__main__':
    run(parser().parse_args())
