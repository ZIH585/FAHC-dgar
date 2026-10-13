"""Run isolation and checkpoints for the pinned author's DGAR protocol."""
import copy
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

PROTOCOL = 'official-dgar-v1'
UPSTREAM_COMMIT = 'c5bbc87467bb22d266b664d30eefb8184c1d3104'


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def rng_state(device):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.device(device).type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.device(device).type == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)


def settings(args):
    excluded = {'gpu', 'run_dir', 'resume', 'test', 'final_validation', 'data_root',
                'num_nodes', 'num_rels'}
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in excluded and not key.startswith('_')}


def fingerprint(paths, root):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        digest.update(b'\0')
    return digest.hexdigest()


def validate_history_cache(train_list, num_relations, directory):
    """Check the existing Original cache without regenerating or sampling it."""
    for stage, triples in enumerate(train_list):
        path = directory / f'snap_{stage}.pkl'
        if not path.is_file():
            raise FileNotFoundError(f'Copy the matching Original history_snap_v2 cache: {path}')
        with path.open('rb') as stream:
            history = pickle.load(stream)
        if not isinstance(history, dict):
            raise ValueError(f'Expected a subject-indexed history dictionary: {path}')
        forward = set(map(tuple, triples.tolist()))
        allowed = forward | {(o, r + num_relations, s) for s, r, o in forward}
        observed = set()
        for subject, rows in history.items():
            for triple in rows:
                row = tuple(triple)
                if len(row) != 3 or str(row[0]) != subject or row not in allowed:
                    raise ValueError(f'History cache has a misindexed/non-training fact at stage {stage}: {path}')
                if row[1] < num_relations:
                    observed.add(row)
        if observed != forward:
            raise ValueError(f'History cache is missing training facts at stage {stage}: {path}')


def prepare_run(args):
    root = Path(__file__).resolve().parents[1]
    data = args.data_root / args.dataset
    files = [data / f'{name}.txt' for name in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        files.append(data / 'e-w-graph.txt')
    args._provenance = {
        'protocol': PROTOCOL,
        'upstream_commit': UPSTREAM_COMMIT,
        'settings': settings(args),
        'source_sha256': fingerprint([p for folder in ('src', 'diffusion', 'rgcn')
                                      for p in (root / folder).glob('*.py')], root),
        'data_sha256': fingerprint(files, data),
        'history_sha256': fingerprint(list((data / 'history_snap_v2').glob('snap_*.pkl')), data),
    }
    if args.run_dir is None:
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not (args.resume or args.test):
        raise ValueError('Use a new --run-dir; use --resume for an existing training run')
    if args.resume and args.resume.resolve().parent != args.run_dir:
        raise ValueError('--resume must be latest.pt inside the same --run-dir')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / 'config.json'
    if args.test:
        saved = json.loads(config_path.read_text())
        if saved != args._provenance:
            raise ValueError('Evaluation config/source/data/cache differ from this run')
    elif not args.resume:
        config_path.write_text(json.dumps(args._provenance, indent=2) + '\n')


def save_training_state(args, stage, model, diffusion, optimizer, diffusion_optimizer,
                        device, auxiliary):
    if args.run_dir is None:
        return
    # model/diffusion are the just-loaded BEST feature-generator weights. Adam
    # is deliberately from the LAST update, not the best epoch. Never pair it
    # with an earlier Adam snapshot: that is a different training algorithm.
    payload = {'version': 2, 'completed_stage': stage, 'provenance': args._provenance,
               'model': cpu_copy(model.state_dict()), 'diffusion': cpu_copy(diffusion.state_dict()),
               'optimizer': cpu_copy(optimizer.state_dict()),
               'diffusion_optimizer': cpu_copy(diffusion_optimizer.state_dict()),
               'rng': rng_state(device), 'auxiliary': auxiliary}
    temporary = args.run_dir / 'latest.pt.tmp'
    torch.save(payload, temporary)
    temporary.replace(args.run_dir / 'latest.pt')


def restore_training_state(args, model, diffusion, teacher, diffusion_teacher,
                           optimizer, diffusion_optimizer, device):
    # Full local training state includes Python/NumPy RNG tuples.
    saved = torch.load(args.resume, map_location='cpu', weights_only=False)
    if saved.get('version') != 2 or saved['provenance'] != args._provenance:
        raise ValueError('Resume config/source/data/cache differ from this checkpoint')
    model.load_state_dict(saved['model'])
    teacher.load_state_dict(saved['model'])
    diffusion.load_state_dict(saved['diffusion'])
    diffusion_teacher.load_state_dict(saved['diffusion'])
    optimizer.load_state_dict(saved['optimizer'])
    diffusion_optimizer.load_state_dict(saved['diffusion_optimizer'])
    # A crash in the next stage may already have overwritten the best files.
    # Restore these as well: the original loop reloads them before training.
    for name, key in (('model.pt', 'model'), ('diffusion.pt', 'diffusion')):
        temporary = args.run_dir / (name + '.tmp')
        torch.save(saved[key], temporary)
        temporary.replace(args.run_dir / name)
    restore_rng(saved['rng'], device)
    return int(saved['completed_stage']) + 1, saved['auxiliary']


def restore_evaluation_rng(args, device, final_stage):
    """Reproduce the stream at entry to final TEST, without an evaluation seed."""
    saved = torch.load(args.run_dir / 'latest.pt', map_location='cpu', weights_only=False)
    if (saved.get('version') != 2 or saved['provenance'] != args._provenance
            or saved['completed_stage'] != final_stage):
        raise ValueError('Final TEST requires a completed run from this baseline protocol')
    # Refuse stale/partially overwritten files rather than pair a saved RNG with
    # different weights. This check does not consume the experiment RNG stream.
    for name, key in (('model.pt', 'model'), ('diffusion.pt', 'diffusion')):
        if key == 'diffusion' and not args.diffuc:
            continue
        weights = torch.load(args.run_dir / name, map_location='cpu', weights_only=True)
        expected = saved[key]
        if weights.keys() != expected.keys() or any(
                not torch.equal(weights[k], expected[k]) for k in expected):
            raise ValueError(f'{name} differs from completed latest.pt; resume the run first')
    restore_rng(saved['rng'], device)
