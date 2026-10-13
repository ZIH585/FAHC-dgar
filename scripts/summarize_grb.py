"""Summarize three completed groups without loading models, data or CUDA."""
import argparse
import csv
import json
from pathlib import Path
import re


PATTERNS = {
    'pooled_mrr': r'^MRR \(filter_ent\): ([0-9.]+)',
    'pooled_hits1': r'^Hits \(filter_ent\) @ 1: ([0-9.]+)',
    'pooled_hits3': r'^Hits \(filter_ent\) @ 3: ([0-9.]+)',
    'pooled_hits10': r'^Hits \(filter_ent\) @ 10: ([0-9.]+)',
    'macro_mrr': r'Average MRR \(macro,[^)]+\): ([0-9.]+)',
    'macro_hits1': r'Average Hits@1 \(macro,[^)]+\): ([0-9.]+)',
    'macro_hits10': r'Average Hits@10 \(macro,[^)]+\): ([0-9.]+)',
    'current_mrr': r'Current MRR \(filter_ent\): ([0-9.]+)',
}


def read_run(directory, group):
    config = json.loads((directory/'config.json').read_text())
    settings = config['settings']
    if settings['grb_mode'] != {'original': 'original', 'half': 'fixed', 'adaptive': 'adaptive'}[group]:
        raise ValueError(f'{group}: incorrect grb-mode')
    if group == 'half' and settings['grb_fixed_weight'] != .5:
        raise ValueError('half requires fixed weight .5')
    log = directory.parent.parent/'logs'/f'{directory.name}.log'
    text = log.read_text(errors='replace')
    if 'Traceback (most recent call last)' in text or 'Final full-pipeline TEST' not in text:
        raise ValueError(f'Run is incomplete or failed: {log}')
    test = text.rsplit('Final full-pipeline TEST', 1)[1]
    test, sep, validation = test.partition('Final full-pipeline VALIDATION')
    if not sep:
        raise ValueError(f'Final validation missing: {log}')
    gpu = re.search(r'Namespace\([^\n]*\bgpu=(-?\d+)', text)
    if not gpu:
        raise ValueError(f'Actual GPU index missing: {log}')
    rows = []
    for split, section in (('TEST', test), ('VALIDATION', validation)):
        row = {'group': group, 'split': split, 'run': str(directory), 'gpu': int(gpu[1])}
        for key, pattern in PATTERNS.items():
            match = re.search(pattern, section, re.MULTILINE)
            if match is None:
                raise ValueError(f'{group}/{split}: missing {key}')
            row[key] = float(match[1])
        rows.append(row)
    balance = []
    if group != 'original':
        path = directory/'grb-stages.jsonl'
        if not path.exists():
            raise ValueError(f'Controller was not logged: {path}')
        # A resume can append a repeated record after the checkpoint was saved.
        # The last completed record for each stage is the relevant one.
        stages = {record['stage']: record for line in path.read_text().splitlines()
                  if (record := json.loads(line))}
        balance = [update for stage in stages.values() for update in stage.get('updates', [])]
        if not balance:
            raise ValueError(f'No replay updates measured: {path}')
    return config, rows, balance


def summarize(runs, output):
    parsed = {group: read_run(path, group) for group, path in runs.items()}
    reference = parsed['original'][0]
    excluded = {'grb_mode'}
    # All knobs other than mode have common defaults in the supplied commands.
    reference_settings = {key: value for key, value in reference['settings'].items() if key not in excluded}
    for group, (config, rows, _) in parsed.items():
        for key in ('protocol', 'upstream_commit', 'source_sha256', 'data_sha256', 'history_sha256'):
            if config[key] != reference[key]:
                raise ValueError(f'{group}: mismatched {key}')
        if {k: v for k, v in config['settings'].items() if k not in excluded} != reference_settings:
            raise ValueError(f'{group}: non-mode training settings differ')
        if rows[0]['gpu'] != parsed['original'][1][0]['gpu']:
            raise ValueError('Use one GPU index sequentially for the three controls')
    rows = [row for _, group_rows, _ in parsed.values() for row in group_rows]
    reference_rows = {row['split']: row for row in parsed['original'][1]}
    half_rows = {row['split']: row for row in parsed['half'][1]}
    for row in rows:
        row['pooled_delta_vs_original_pp'] = 100*(row['pooled_mrr']-reference_rows[row['split']]['pooled_mrr'])
        row['macro_delta_vs_original_pp'] = 100*(row['macro_mrr']-reference_rows[row['split']]['macro_mrr'])
        row['pooled_delta_vs_half_pp'] = 100*(row['pooled_mrr']-half_rows[row['split']]['pooled_mrr'])
        row['macro_delta_vs_half_pp'] = 100*(row['macro_mrr']-half_rows[row['split']]['macro_mrr'])
    weights = {}
    for group, (_, _, updates) in parsed.items():
        if updates:
            values = [row['weight'] for row in updates]
            valid = [row for row in updates if row['valid_signal']]
            weights[group] = {'updates': len(values), 'mean': sum(values)/len(values),
                'min': min(values), 'max': max(values),
                'fraction_at_half': sum(abs(v-.5) < 1e-6 for v in values)/len(values),
                'fraction_at_one': sum(abs(v-1) < 1e-6 for v in values)/len(values),
                'negative_cosine_fraction': sum(row['cosine'] < 0 for row in valid)/len(valid) if valid else None}
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new summary directory')
    with (output/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps({'metrics': rows, 'weights': weights,
        'provenance': reference, 'limits': 'One seed is a screening experiment; GPU index alone does not establish identical hardware/software.'}, indent=2)+'\n')
    print('| Group | Split | Pooled MRR % | Macro MRR % | Δ Original pp | Δ Half pp |')
    print('|---|---|---:|---:|---:|---:|')
    for row in rows:
        print(f"| {row['group']} | {row['split']} | {100*row['pooled_mrr']:.4f} | "
              f"{100*row['macro_mrr']:.4f} | {row['pooled_delta_vs_original_pp']:+.4f} | "
              f"{row['pooled_delta_vs_half_pp']:+.4f} |")
    print(json.dumps(weights, indent=2))
    return rows, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for group in ('original', 'half', 'adaptive'):
        parser.add_argument(f'--{group}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize({group: getattr(args, group).resolve() for group in ('original', 'half', 'adaptive')}, args.output)
