"""Small state helpers for reproducible FAHC probing and evaluation."""

from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch


@contextmanager
def isolated_rng(seed: int, device: torch.device | str = "cpu") -> Iterator[None]:
    """Run with a deterministic RNG stream, then restore all touched streams.

    Python, NumPy, and the PyTorch CPU default generator are always isolated.
    For a CUDA device, only that device's generator is seeded and restored.
    A CPU call never invokes a CUDA RNG API. State is restored on exceptions.
    """
    requested = torch.device(device)
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("isolated_rng supports CPU or CUDA devices")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_index = None
    cuda_device = None
    cuda_state = None
    if requested.type == "cuda":
        cuda_index = (requested.index if requested.index is not None
                      else torch.cuda.current_device())
        cuda_device = torch.device("cuda", cuda_index)
        cuda_state = torch.cuda.get_rng_state(cuda_device)

    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # torch.manual_seed also calls CUDA seeding; use the CPU generator
        # directly so no other GPU's stream changes.
        torch.default_generator.manual_seed(seed)
        if cuda_index is not None:
            torch.cuda.default_generators[cuda_index].manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_device is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {copy.deepcopy(key): _clone_cpu(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        items = tuple(_clone_cpu(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, set):
        return {_clone_cpu(item) for item in value}
    return copy.deepcopy(value)


def cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    """Return an independent CPU snapshot of an optimizer's state dict.

    Every nested tensor is detached and cloned, including CPU tensors whose
    ``.cpu()`` method would otherwise return the original object.
    """
    return _clone_cpu(optimizer.state_dict())
