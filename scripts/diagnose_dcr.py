"""Paired DCR endpoint screen on a completed Original checkpoint, VALIDATION only."""
import argparse
from contextlib import contextmanager
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
from diffusion import dcr
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import (PROTOCOL, UPSTREAM_COMMIT, fingerprint,
                         validate_history_cache, rng_state, restore_rng)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed Original checkpoint, read-only')
    p.add_argument('--output', type=Path, required=True, help='Fresh directory outside the input run')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--stages', default='60,120,180,240,300')
    p.add_argument('--seed', type=int, default=1, help='Diagnostic HCP/noise seed, not a new model seed')
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
    # Optional explicit mean checkpoint support is used only by the independent
    # scale diagnostic. The original endpoint screen retains its default guard.
    checkpoint_mode = getattr(options, 'checkpoint_mode', 'original')
    if checkpoint_mode not in ('original', 'mean'):
        raise ValueError('Only audited Original/mean checkpoints are supported')
    expected_protocol = PROTOCOL if checkpoint_mode == 'original' else dcr.PROTOCOL
    local_hash = fingerprint([p for folder in ('src', 'diffusion', 'rgcn')
                              for p in (ROOT / folder).glob('*.py')], ROOT)
    sources = {'6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538',
               'c22dc00022ff00ef8ebf9486d32f839ef9f60b5e0c59d04b79413401ba96b3d5', local_hash}
    if (config['protocol'] != expected_protocol or config['upstream_commit'] != UPSTREAM_COMMIT
            or config['source_sha256'] not in sources
            or s.get('grb_mode', 'original') != 'original'
            or s.get('dcr_mode', 'original') != checkpoint_mode
            or any(k.startswith(('fahc', 'ecri', 'qcdr', 'bcr')) for k in s)
            or not s['diffuc'] or s['delete_feature_reply'] or s['delete_his_prompt']
            or s['delete_score'] or s['vl']):
        raise ValueError('Use a completed audited checkpoint matching checkpoint_mode')
    args = build_arg_parser().parse_args([])
    for k, v in s.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.data_root, args.gpu = options.data_root.resolve(), options.gpu
    if checkpoint_mode == 'mean':
        args.run_dir = run_dir  # Required by CLI validation; loader never writes.
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


def buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


@torch.no_grad()
def restore_buffers(model, saved):
    for name, value in model.named_buffers():
        value.copy_(saved[name])


@contextmanager
def stage_state(model, generator, device, seed):
    """Diagnostic-only isolation; the real training CLI retains shared RNG."""
    old_rng = rng_state(device)
    old_buffers = [(module, buffers(module)) for module in (model, generator)]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        for module, saved in old_buffers:
            restore_buffers(module, saved)
        restore_rng(old_rng, device)
        dcr.release_guidance_graph(model)


@torch.no_grad()
def paired_endpoints(generator, model, args, tags, guide_history, static, stage, cuda):
    """One network prediction shared by all four arms, including its dropout.

    Capture the pre-guidance RNG/buffers, then re-evaluate guidance at x0. The
    historical tags are training facts; held-out queries/answers are absent.
    """
    if args.dcr_mode != 'original' or args.vl or args.delete_score:
        raise ValueError('Paired endpoint capture requires the Original guided replay path')
    if not len(tags):
        return {mode: None for mode in dcr.MODES}, dict(facts=0)
    device = tags.device
    captured = {}
    sampler = generator.diffu
    original_mean = sampler.p_mean_variance

    def capture_x0(module, inputs, output):
        if 'x0' in captured:
            raise ValueError('Expected exactly one denoiser forward per replay')
        captured['x0'] = output.detach().clone()
        captured['guide_rng'] = rng_state(device)
        captured['guide_buffers'] = buffers(model)

    def capture_mean(*a, **kw):
        result = original_mean(*a, **kw)
        captured['mean'] = result[0].detach().clone()
        captured['t'] = a[4].detach().clone()
        return result

    handle = sampler.xstart_model.register_forward_hook(capture_x0)
    had_override = 'p_mean_variance' in sampler.__dict__
    old_override = sampler.__dict__.get('p_mean_variance')
    sampler.p_mean_variance = capture_mean
    try:
        original = generator(None, tags, args, False, cuda, static_graph=static,
            ct=stage, model=model, history_glist=guide_history, triples=None)[1].detach()
    finally:
        handle.remove()
        if had_override:
            sampler.p_mean_variance = old_override
        else:
            del sampler.p_mean_variance
    after_rng = rng_state(device)
    after_buffers = buffers(model)
    try:
        restore_rng(captured['guide_rng'], device)
        restore_buffers(model, captured['guide_buffers'])
        x0 = captured['x0']
        tail, gradient = dcr.guided_tail(model, x0, x0[:, -1], tags,
            guide_history, static, cuda, args)
    finally:
        restore_rng(after_rng, device)
        restore_buffers(model, after_buffers)
        dcr.release_guidance_graph(model)
    mean = torch.cat((x0[:, :2], captured['mean'].unsqueeze(1)), dim=1)
    guided = torch.cat((x0[:, :2], tail.unsqueeze(1)), dim=1)
    noise_effect = original[:, -1] - mean[:, -1]
    scalars = torch.stack((x0[:, 0].norm(dim=-1).mean(), x0[:, -1].norm(dim=-1).mean(),
        mean[:, -1].norm(dim=-1).mean(), noise_effect.norm(dim=-1).mean(),
        (args.classifier_scale * gradient).norm(dim=-1).mean(), tail.norm(dim=-1).mean())).cpu().tolist()
    names = ('x0_head_norm', 'x0_tail_norm', 'posterior_guided_tail_norm',
             'terminal_noise_effect_norm', 'x0_scaled_guidance_norm', 'x0_guided_tail_norm')
    statistics = dict(facts=len(tags), denoiser_calls=1, norms=dict(zip(names, scalars)),
        posterior_x0_coefficient=float(sampler.posterior_mean_coef1[-1]),
        terminal_noise_std=float(np.sqrt(sampler.betas[-1])),
        original_fusion_alpha=float(model.alpha.detach().cpu()))
    return dict(original=original, mean=mean, x0=x0, **{'guided-x0': guided}), statistics


def evaluation_history(train, t, args):
    # Match the author's conditional, including nondefault lengths.
    return train[:t] if t - args.train_history_len < 0 else train[t-args.test_history_len:t]


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
    answers = utils.load_all_answers_for_time_filter(validation, args.num_rels, args.num_nodes, False)
    rows, generation = [], []
    for t in stages:
        if not len(validation[t]):
            continue
        rng = random.Random(options.seed + 1000003 * t)
        times = list(range(t)) if t - 1 < args.reply_batch else rng.sample(list(range(t)), args.reply_batch)
        noise_seed = rng.getrandbits(63)
        facts = set()
        for i in times:
            with (args.data_root / args.dataset / 'history_snap_v2' / f'snap_{i}.pkl').open('rb') as stream:
                cache = pickle.load(stream)
            for s in set(map(int, validation[t][:, 0])):
                facts.update(tuple(map(int, x)) for x in cache.get(str(s), ()))
        tags = torch.as_tensor(list(facts), dtype=torch.long, device=device).reshape(-1, 3)
        triples = torch.as_tensor(validation[t], dtype=torch.long, device=device)
        history = [utils.build_sub_graph(args.num_nodes, args.num_rels, x, cuda, args.gpu)
                   for x in evaluation_history(train, t, args)]
        guide = [utils.build_sub_graph(args.num_nodes, args.num_rels, x, cuda, args.gpu)
                 for x in evaluation_history(train, t - 1, args)]
        start = time.monotonic()
        with stage_state(model, generator, device, noise_seed):
            variants, stats = paired_endpoints(generator, model, args, tags, guide, static, t, cuda)
            ranks, feature_norms = {}, {}
            with torch.no_grad():
                for mode, features in variants.items():
                    expanded, scores, _ = model.predict(history, args.num_rels, static,
                        triples, features, features[:, -1] if features is not None else None, tags, cuda)
                    ranks[mode] = utils.get_total_rank(expanded, scores.clone(), answers[t], eval_bz=1000)[3].cpu().tolist()
                    if features is not None:
                        ids, bank = model.select_entity(tags, features, cuda)
                        feature_norms[mode] = float(bank[ids].norm(dim=-1).mean().cpu())
                    else:
                        feature_norms[mode] = None
                    del scores
        stats.update(stage=t, times=times, noise_seed=noise_seed,
                     aggregate_entity_norm=feature_norms, seconds=time.monotonic()-start)
        generation.append(stats)
        for i, (s, r, o) in enumerate(expanded.cpu().tolist()):
            rows.append(dict(stage=t, subject=s, relation=r, target=o,
                direction='forward' if i < len(triples) else 'inverse',
                **{mode + '_rank': int(ranks[mode][i]) for mode in dcr.MODES}))
        print(f'Stage {t}: facts={len(tags)}, validation_queries={len(expanded)}; '
            + ', '.join(f'{m} MRR={metrics([r for r in rows if r["stage"]==t], m+"_rank")["mrr"]:.6f}' for m in dcr.MODES), flush=True)
        del variants
    if not rows:
        raise ValueError('No validation queries')
    overall = {m: metrics(rows, m+'_rank') for m in dcr.MODES}
    by_stage = {str(t): {m: metrics([r for r in rows if r['stage']==t], m+'_rank') for m in dcr.MODES}
                for t in stages if any(r['stage']==t for r in rows)}
    delta = {m: overall['guided-x0']['mrr']-overall[m]['mrr'] for m in ('original', 'mean', 'x0')}
    good_stages = sum(x['guided-x0']['mrr'] >= x['original']['mrr']
                      and x['guided-x0']['mrr'] >= x['mean']['mrr'] for x in by_stage.values())
    changes = {}
    for reference in ('original', 'mean', 'x0'):
        differences = [1/r['guided-x0_rank']-1/r[reference+'_rank'] for r in rows]
        changes[reference] = dict(improved=sum(x>0 for x in differences),
            worsened=sum(x<0 for x in differences), unchanged=sum(x==0 for x in differences))
    decision = dict(delta_mrr=delta, stages_not_worse_than_original_and_mean=good_stages,
        provisional_gate=delta['original'] >= .003 and delta['mean'] >= .003
                         and delta['x0'] >= 0 and good_stages >= 4,
        note='Resource screen, not statistical significance; inspect direction damage and query concentration.')
    summary = dict(protocol='paired-dcr-validation-v1', training_provenance=config,
        diagnostic_source_sha256=source_hash,
        diagnostic_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        overall=overall, by_stage=by_stage,
        by_direction={d:{m:metrics([r for r in rows if r['direction']==d], m+'_rank') for m in dcr.MODES}
                      for d in ('forward','inverse')}, generation=generation, changes=changes, decision=decision,
        limits='Final Original model; selected slices FULL VALIDATION, same HCP and one shared denoiser output. Diagnostic RNG/buffers isolated; not the full author evaluation RNG trajectory, paper Average, or chronological forgetting. One HCP/noise realization. No optimization; training benefit untested.')
    with (output/'queries.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'generation.jsonl').open('w') as stream:
        for data in generation:
            stream.write(json.dumps(data,sort_keys=True)+'\n')
    print(json.dumps(dict(overall=overall, decision=decision),indent=2), flush=True)
    print(f'Completed without training. Outputs: {output}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
