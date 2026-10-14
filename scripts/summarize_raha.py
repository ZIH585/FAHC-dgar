"""Recompute paired RAHA results from CSV using only the Python standard library."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def metric(rows):
    ranks = [int(r['rank_filtered']) for r in rows]
    if not ranks:
        return {'queries': 0, 'mrr': None, 'hits1': None, 'hits3': None, 'hits10': None}
    if any(r < 1 for r in ranks):
        raise ValueError('invalid ranks')
    return {'queries': len(ranks), 'mrr': statistics.fmean(1 / r for r in ranks),
            **{f'hits{k}': statistics.fmean(r <= k for r in ranks) for k in (1, 3, 10)}}


def audit_split(run, split, config):
    candidates = sorted((run / 'diagnostics').glob(f'{split}-*/summary.json'))
    if not candidates:
        raise ValueError(f'{run}: no completed {split} diagnostics')
    path = candidates[-1]
    summary = json.loads(path.read_text())
    if summary['provenance'] != config or summary['split'] != split:
        raise ValueError(f'{path}: provenance/split mismatch')
    with (path.parent / 'queries.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    final = summary['final_stage']
    stages = {int(r['stage']) for r in rows}
    if stages != set(range(1, final + 1)):
        raise ValueError(f'{path}: incomplete CSV stage coverage')
    groups = {}
    for row in rows:
        groups.setdefault(int(row['stage']), []).append(row)
    calculated = {'macro_mrr': statistics.fmean(metric(groups[s])['mrr'] for s in range(1, final)) if final > 1 else None,
                  'pooled': metric(rows), 'current': metric(groups[final]),
                  'groups': {name: metric([r for r in rows if int(r[key]) == val])
                             for name, key, val in (('forward', 'inverse', 0), ('inverse', 'inverse', 1),
                                                    ('repeated', 'seen_before', 1), ('unseen', 'seen_before', 0))}}
    def check(left, right):
        if isinstance(left, dict):
            for key in left:
                check(left[key], right[key])
        elif left is None:
            if right is not None:
                raise ValueError(f'{path}: empty group mismatch')
        elif not math.isclose(left, right, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'{path}: summary/CSV mismatch: {left} != {right}')
    check(calculated, summary)
    identities = [[r[k] for k in ('stage', 'subject', 'relation', 'target', 'inverse', 'seen_before')] for r in rows]
    calculated['query_sha256'] = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    return calculated


def summarize(runs, output):
    results, cohort = [], None
    for directory in runs:
        run = directory.resolve()
        config = json.loads((run / 'config.json').read_text())
        scientific = {k: v for k, v in config['settings'].items() if k not in ('raha_mode', 'raha_rank')}
        identity = (config['source_sha256'], config['data_sha256'], config['history_sha256'], scientific)
        if cohort is not None and identity != cohort:
            raise ValueError('runs differ in source, data, history cache or backbone settings')
        cohort = identity
        mode = config['settings'].get('raha_mode', 'original')
        expected_protocol = 'official-dgar-v1' if mode == 'original' else f'raha-{mode}-v1'
        if config['protocol'] != expected_protocol:
            raise ValueError('unexpected method/protocol')
        result = {'run': str(run), 'mode': mode, 'rank': config['settings'].get('raha_rank', 0)}
        for split in ('validation', 'test'):
            metrics = audit_split(run, split, config)
            result[split] = metrics
        results.append(result)
    if sum(r['mode'] == 'original' for r in results) != 1:
        raise ValueError('include exactly one Original control')
    original = next(r for r in results if r['mode'] == 'original')
    for result in results:
        for split in ('validation', 'test'):
            if result[split]['query_sha256'] != original[split]['query_sha256']:
                raise ValueError('query identities/order/history labels differ between runs')
    flat = []
    for result in results:
        for split in ('validation', 'test'):
            m = result[split]
            flat.append({'run': result['run'], 'mode': result['mode'], 'rank': result['rank'], 'split': split,
                         'macro_mrr': m['macro_mrr'], 'macro_gain_pp': 100 * (m['macro_mrr'] - original[split]['macro_mrr']),
                         **{'pooled_' + k: v for k, v in m['pooled'].items()}, 'current_mrr': m['current']['mrr'],
                         **{name + '_mrr': v['mrr'] for name, v in m['groups'].items()}})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    with (output / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for row in flat:
        print(f"{row['split']:10} {row['mode']:8} r{row['rank']:<2} macro={100 * row['macro_mrr']:.4f}% "
              f"gain={row['macro_gain_pp']:+.4f}pp pooled={100 * row['pooled_mrr']:.4f}%")
    print(f'Wrote {output / "metrics.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Completed RAHA run directories, including Original')
    parser.add_argument('--output', required=True, type=Path, help='Fresh summary directory')
    options = parser.parse_args()
    summarize(options.runs, options.output)
