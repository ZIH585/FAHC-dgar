"""Read completed BCR experiment logs/configs; no models, torch or GPU."""
import argparse
import csv
import json
from pathlib import Path
import re

PATTERNS = {
    'pooled_mrr': r'^MRR \(filter_ent\): ([0-9.]+)',
    'hits1': r'^Hits \(filter_ent\) @ 1: ([0-9.]+)',
    'hits3': r'^Hits \(filter_ent\) @ 3: ([0-9.]+)',
    'hits10': r'^Hits \(filter_ent\) @ 10: ([0-9.]+)',
    'macro_mrr': r'Average MRR \(macro,[^)]+\): ([0-9.]+)',
    'current_mrr': r'Current MRR \(filter_ent\): ([0-9.]+)',
}


def read(directory, mode):
    config = json.loads((directory / 'config.json').read_text())
    if config['settings']['bcr_mode'] != mode:
        raise ValueError(f'Wrong mode in {directory}')
    expected = 'official-dgar-v1' if mode == 'original' else 'bcr-directional-v1'
    if config['protocol'] != expected:
        raise ValueError(f'Unexpected protocol in {directory}')
    logfile = directory.parent.parent / 'logs' / f'{directory.name}.log'
    text = logfile.read_text(errors='replace')
    if 'Traceback (most recent call last)' in text:
        raise ValueError(f'Failed run: {logfile}')
    if 'Final full-pipeline TEST' not in text:
        raise ValueError(f'Incomplete run: {logfile}')
    test, sep, validation = text.rsplit('Final full-pipeline TEST', 1)[1].partition('Final full-pipeline VALIDATION')
    if not sep:
        raise ValueError(f'Final validation missing: {logfile}')
    gpu = re.search(r'Namespace\([^\n]*\bgpu=(-?\d+)', text)
    if gpu is None:
        raise ValueError(f'GPU index missing: {logfile}')
    rows = []
    for split, section in [('TEST', test), ('VALIDATION', validation)]:
        values = dict(group=mode, split=split, run=str(directory), gpu=int(gpu[1]))
        for key, pattern in PATTERNS.items():
            match = re.search(pattern, section, re.MULTILINE)
            if not match:
                raise ValueError(f'{logfile}: missing {split}/{key}')
            values[key] = float(match[1])
        rows.append(values)
    stats = {}
    if mode != 'original':
        # A resumed partial stage may leave duplicate appended records.
        train = {}
        for line in (directory / 'bcr.jsonl').read_text().splitlines():
            record = json.loads(line)
            if record['phase'] == 'train':
                train[record['stage']] = record
        contexts = [c for r in train.values() for c in r['contexts']]
        if not contexts:
            raise ValueError(f'BCR was never logged: {directory}')
        stats = dict(stages_logged=len(train), contexts=len(contexts),
            total_selected=sum(c['selected'] for c in contexts),
            total_budget=sum(c['budget'] for c in contexts),
            selected_incoming_only=sum(c['selected_incoming_only'] for c in contexts))
    return config, rows, stats


def summarize(runs, output):
    parsed = {mode: read(path.resolve(), mode) for mode, path in runs.items()}
    reference = parsed['original'][0]
    reference_settings = {k: v for k, v in reference['settings'].items() if k != 'bcr_mode'}
    for mode, (config, rows, _) in parsed.items():
        for key in ('source_sha256', 'upstream_commit', 'data_sha256', 'history_sha256'):
            if config[key] != reference[key]:
                raise ValueError(f'{mode}: mismatched {key}; use same branch/data/cache')
        if {k: v for k, v in config['settings'].items() if k != 'bcr_mode'} != reference_settings:
            raise ValueError(f'{mode}: non-mode settings differ')
        if rows[0]['gpu'] != parsed['original'][1][0]['gpu']:
            raise ValueError('Use the same GPU index sequentially')
    refs = {r['split']: r for r in parsed['original'][1]}
    rows = [r for _, rs, _ in parsed.values() for r in rs]
    for r in rows:
        r['pooled_delta_original_pp'] = 100 * (r['pooled_mrr'] - refs[r['split']]['pooled_mrr'])
        r['macro_delta_original_pp'] = 100 * (r['macro_mrr'] - refs[r['split']]['macro_mrr'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh summary directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(metrics=rows, replay={m: x[2] for m, x in parsed.items()},
        provenance={m: x[0] for m, x in parsed.items()},
        limits='OFF is official DGAR. Enabled arms explicitly separate directions and isolate generator buffers/RNG. Query budgets match between enabled controls; total inference cost is larger than Original. Same GPU index alone does not prove identical software/hardware.')
    (output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    for r in rows:
        print(f'{r["group"]}/{r["split"]}: pooled={100*r["pooled_mrr"]:.4f}%, '
              f'macro={100*r["macro_mrr"]:.4f}%, delta={r["pooled_delta_original_pp"]:+.4f}pp')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--incident', type=Path, required=True)
    p.add_argument('--outgoing', type=Path)
    p.add_argument('--random', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summarize({m: getattr(args, m) for m in ('original', 'incident', 'outgoing', 'random')
               if getattr(args, m) is not None}, args.output.resolve())
