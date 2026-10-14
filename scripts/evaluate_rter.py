"""Paired RTER scores on an Original checkpoint; no training, one DGAR forward.

Validation follows the author's full TEST RNG/buffer trajectory first. All
evidence controls share the exact same Original logits, full candidate set,
filtered answers, generated features and decoder batch. No target enters the
evidence computation. Hyperparameter selection uses development VALIDATION.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rgcn import utils
from src import main, rter
from src.runtime import restore_rng, fingerprint
from rter_checkpoint import load_original


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True, help='Completed Original run, read-only')
    p.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    p.add_argument('--data-root', type=Path, default=Path('/data1/zihao/DGAR-code/data'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--stages', default='all', help='all or increasing comma-separated snapshot indices')
    p.add_argument('--betas', default='0.1,0.3,1', help='Small prespecified grid; TEST requires one beta')
    p.add_argument('--modes', default='transition,repeat,popularity,shuffled')
    p.add_argument('--selection-file', type=Path, help='Lock TEST settings to a successful VALIDATION summary')
    p.add_argument('--development-end', type=int, default=None, help='Default midpoint of timeline; selected before seeing scores')
    p.add_argument('--rter-horizon', type=float, default=30.)
    p.add_argument('--rter-smoothing', type=float, default=20.)
    p.add_argument('--rter-window', type=int, default=30)
    p.add_argument('--rter-clip', type=float, default=3.)
    p.add_argument('--rter-channel', choices=('cross', 'all'), default='cross')
    p.add_argument('--rter-shuffle-seed', type=int, default=1)
    p.add_argument('--check-only', action='store_true', help='Load/validate, stop before inference')
    return p


def column(mode, beta):
    return f'{mode}_b{beta:g}_rank'


def metrics(rows, key, last_stage):
    if not rows:
        return None
    ranks = np.asarray([r[key] for r in rows], dtype=float)
    by_time = {}
    for row in rows:
        by_time.setdefault(row['stage'], []).append(1./row[key])
    past = [np.mean(values) for t, values in by_time.items() if t != last_stage]
    return dict(queries=len(rows), pooled_mrr=float(np.mean(1/ranks)),
                macro_mrr=float(np.mean(past)) if past else None,
                hits1=float(np.mean(ranks<=1)), hits3=float(np.mean(ranks<=3)),
                hits10=float(np.mean(ranks<=10)))


def summarize(rows, modes, betas, last_stage, cut, split):
    keys = ['original_rank']+[column(m, b) for m in modes for b in betas]
    summary = dict(overall={k:metrics(rows, k, last_stage) for k in keys},
        by_stage={str(t):{k:metrics([r for r in rows if r['stage']==t], k, last_stage) for k in keys}
                  for t in sorted({r['stage'] for r in rows})},
        by_direction={d:{k:metrics([r for r in rows if r['direction']==d], k, last_stage) for k in keys}
                      for d in ('forward', 'inverse')},
        by_fact_type={g:{k:metrics([r for r in rows if r['seen_before']==v], k, last_stage) for k in keys}
                      for g,v in (('repeated',True), ('unseen',False))})
    if split == 'test':
        summary['selection'] = {'performed':False, 'note':'TEST uses a locked single beta; no selection'}
        return summary
    dev = [r for r in rows if r['stage'] <= cut and r['stage'] != last_stage]
    confirm = [r for r in rows if cut < r['stage'] < last_stage]
    if not dev or not confirm:
        summary['selection'] = {'performed':False, 'note':'Both development and confirmation VALIDATION are required'}
        return summary
    selected = {}
    for m in modes:
        beta = max(betas, key=lambda b:(metrics(dev, column(m,b), last_stage)['macro_mrr'], -b))
        key = column(m, beta)
        changes = [1/r[key]-1/r['original_rank'] for r in confirm]
        gains = sorted((v for v in changes if v > 0), reverse=True)
        selected[m] = dict(beta=beta, development=metrics(dev, key, last_stage),
            confirmation=metrics(confirm, key, last_stage),
            directions={d:metrics([r for r in confirm if r['direction']==d], key, last_stage)
                        for d in ('forward', 'inverse')},
            improved=sum(v>0 for v in changes), worsened=sum(v<0 for v in changes),
            unchanged=sum(v==0 for v in changes), net_rr_gain=float(sum(changes)),
            top5_positive_rr_gain=float(sum(gains[:5])))
    reference = metrics(confirm, 'original_rank', last_stage)
    gate = False
    if 'transition' in selected:
        x = selected['transition']['confirmation']
        delta_macro = x['macro_mrr']-reference['macro_mrr']
        delta_pooled = x['pooled_mrr']-reference['pooled_mrr']
        gate = delta_macro >= .003 and delta_pooled >= .003
        for control in ('repeat', 'popularity', 'shuffled'):
            gate = gate and control in selected and x['macro_mrr'] > selected[control]['confirmation']['macro_mrr']
        for d in ('forward', 'inverse'):
            baseline = metrics([r for r in confirm if r['direction']==d], 'original_rank', last_stage)
            method = selected['transition']['directions'][d]
            gate = gate and baseline is not None and method is not None and method['pooled_mrr']-baseline['pooled_mrr'] >= -.003
    summary['selection'] = dict(performed=True, development_end=cut,
        confirmation_original=reference, selected=selected, provisional_gate=bool(gate),
        note='Resource gate, not statistical significance. Check unseen-fact gain and concentration before proceeding. Previously examined VALIDATION is exploratory.')
    return summary


def run(options):
    if options.gpu < -1:
        raise ValueError('Invalid GPU')
    locked = None
    if options.selection_file is not None:
        locked = json.loads(options.selection_file.read_text())
        selection = locked.get('selection', {})
        if (options.split != 'test' or locked.get('protocol') != 'paired-rter-score-v1'
                or locked.get('options', {}).get('split') != 'validation'
                or not selection.get('performed') or not selection.get('provisional_gate')):
            raise ValueError('selection-file requires a VALIDATION summary passing its prespecified gate')
        options.betas = str(selection['selected']['transition']['beta'])
        options.modes = 'transition'
        for name in ('rter_horizon', 'rter_smoothing', 'rter_window', 'rter_clip', 'rter_channel', 'rter_shuffle_seed'):
            setattr(options, name, locked['options'][name])
    betas = sorted(set(float(x) for x in options.betas.split(',')))
    modes = options.modes.split(',')
    if not betas or any(not math.isfinite(b) or b <= 0 for b in betas):
        raise ValueError('Positive finite betas required')
    if not modes or len(set(modes)) != len(modes) or any(m not in rter.MODES[1:] for m in modes):
        raise ValueError('Invalid evidence modes')
    if options.split == 'test' and len(betas) != 1:
        raise ValueError('Lock one beta before TEST; tune on VALIDATION only')
    output, source = options.output.resolve(), options.run_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the input checkpoint directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    loaded = load_original(options)
    args, config, source_hash, train, valid, test, static, model, generator, device, cuda, rng = loaded
    if locked is not None and (locked['training_provenance'] != config or locked['source_sha256'] != source_hash):
        raise ValueError('Locked selection belongs to a different model/data/cache or RTER implementation')
    index = rter.RelationEvidence(train, args.num_nodes, args.num_rels, options.rter_horizon,
        options.rter_smoothing, options.rter_window, options.rter_clip,
        options.rter_channel, options.rter_shuffle_seed)
    last = len(train)-1
    stages = set(range(1,len(train))) if options.stages == 'all' else set(int(x) for x in options.stages.split(','))
    if not stages or min(stages)<1 or max(stages)>last:
        raise ValueError('Selected stages outside evaluated timeline')
    cut = options.development_end if options.development_end is not None else len(train)//2
    if not 1 <= cut < last:
        raise ValueError('Development end must leave confirmation stages')
    if options.check_only:
        print(f'Validated Original: {len(train)} snapshots, N={args.num_nodes}, R={args.num_rels}; no inference/training.')
        return
    output.mkdir(parents=True)
    answers = {name:utils.load_all_answers_for_time_filter(slices, args.num_rels, args.num_nodes, False)
               for name,slices in (('test',test),('validation',valid))}
    rel_answers = {name:utils.load_all_answers_for_time_filter(slices, args.num_rels, args.num_nodes, True)
                   for name,slices in (('test',test),('validation',valid))}
    rows, evidence_log, stage = [], [], {}
    base_predict, base_progress = model.predict, main.tqdm
    phase = None

    def progress(items):
        for t, snap in enumerate(base_progress(items)):
            stage.update(t=t, snap=snap)
            yield snap

    def paired_predict(*a, **kw):
        result = base_predict(*a, **kw)
        t = stage['t']
        if phase != options.split or t not in stages:
            return result
        expanded, scores, _ = result
        start = time.monotonic()
        known = rter.known_queries(stage['snap'], args.num_rels)
        index.advance(t)
        ranks = {'original_rank':utils.get_total_rank(expanded, scores.clone(), answers[phase][t], eval_bz=1000)[3].cpu().tolist()}
        for mode in modes:
            evidence = index.evidence(known, mode)
            evidence_log.append(dict(protocol=rter.PROTOCOL, phase=phase, **evidence.statistics))
            for beta in betas:
                modified = evidence.apply(scores, beta)
                # Author filtering mutates its input. Empty evidence may return
                # the Original tensor itself, so always protect shared logits.
                ranks[column(mode,beta)] = utils.get_total_rank(expanded, modified.clone(), answers[phase][t], eval_bz=1000)[3].cpu().tolist()
        triples = expanded.detach().cpu().tolist()  # Offline labels for reporting only.
        for i,(s,r,o) in enumerate(triples):
            rows.append(dict(stage=t, subject=s, relation=r, target=o,
                direction='forward' if i<len(stage['snap']) else 'inverse',
                seen_before=(r,o) in index.events.get(s,{}), **{k:int(v[i]) for k,v in ranks.items()}))
        print(f'RTER stage={t} queries={len(triples)} evidence/ranking_seconds={time.monotonic()-start:.3f}', flush=True)
        return result  # Original scores returned to main; its trajectory is preserved.

    model.predict, main.tqdm = paired_predict, progress
    restore_rng(rng, device)
    guide = [utils.build_sub_graph(args.num_nodes, args.num_rels, x, cuda, args.gpu)
             for x in train[max(0,last-args.train_history_len):last]]
    final = torch.as_tensor(train[last], device=device, dtype=torch.long)
    start = time.monotonic()
    try:
        phase = 'test'
        print('Original full TEST trajectory (also warms VALIDATION RNG/buffers)',flush=True)
        result = main.test(args, model, train, test, args.num_rels,args.num_nodes,cuda,
            answers['test'],rel_answers['test'],source/'model.pt',static,'test',guide,
            [row for row in final],static,generator)
        if options.split == 'validation':
            phase = 'validation'
            print('Original full VALIDATION trajectory; paired RTER scores',flush=True)
            main.test(args,model,train,valid,args.num_rels,args.num_nodes,cuda,
                answers['validation'],rel_answers['validation'],source/'model.pt',static,'test',
                result[4],result[5],static,generator)
    finally:
        model.predict, main.tqdm = base_predict, base_progress
    if not rows:
        raise ValueError('No selected queries')
    summary = summarize(rows,modes,betas,last,cut,options.split)
    summary.update(protocol='paired-rter-score-v1',training_provenance=config,
        source_sha256=source_hash,
        analysis_sha256=fingerprint([*list((ROOT/'scripts').glob('*.py')), ROOT/'src/rter.py'],ROOT),
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(options).items()},
        seconds=time.monotonic()-start,
        limits='Frozen final Original checkpoint, author retrospective evaluation trajectory. RTER evidence uses train[:t] only. Methods share logits; no optimization. Sampled stages are not paper Average. A small beta grid uses development VALIDATION, not TEST.')
    with (output/'queries.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'rter.jsonl').open('w') as stream:
        for entry in evidence_log:
            stream.write(json.dumps(entry,sort_keys=True)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(overall=summary['overall'],selection=summary['selection']),indent=2),flush=True)
    print(f'Completed without training. Outputs: {output}',flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
