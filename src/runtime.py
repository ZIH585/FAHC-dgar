"""Shared reproducibility and stage checkpoint helpers for Original DGAR."""
import copy
import hashlib
import json
import pickle
import random
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import torch


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


@contextmanager
def isolated_rng(seed, device):
    state = rng_state(device)
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.default_generator.manual_seed(seed)
        if torch.device(device).type == 'cuda':
            torch.cuda.default_generators[torch.device(device).index].manual_seed(seed)
        yield
    finally:
        restore_rng(state, device)


def deterministic_evaluation(function):
    @wraps(function)
    def wrapped(args, model, *positional, **keywords):
        with isolated_rng(args.seed + 7000003, next(model.parameters()).device):
            return function(args, model, *positional, **keywords)
    return wrapped


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
    payload = {'version': 1, 'completed_stage': stage, 'provenance': args._provenance,
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
    if saved.get('version') != 1 or saved['provenance'] != args._provenance:
        raise ValueError('Resume config/source/data/cache differ from this checkpoint')
    model.load_state_dict(saved['model'])
    teacher.load_state_dict(saved['model'])
    diffusion.load_state_dict(saved['diffusion'])
    diffusion_teacher.load_state_dict(saved['diffusion'])
    optimizer.load_state_dict(saved['optimizer'])
    diffusion_optimizer.load_state_dict(saved['diffusion_optimizer'])
    restore_rng(saved['rng'], device)
    return int(saved['completed_stage']) + 1, saved['auxiliary']
