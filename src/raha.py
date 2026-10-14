"""Role-aware historical adaptation, before Original DGAR entity aggregation.

No sampler, teacher, cache, extra loss, or temporal statistics live here.
Head/tail refer to the positions in each stored historical triple, including
inverse triples if the existing HCP cache contains them.
"""
import copy

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """A(z) = z + up(down(z)); no bias, dropout, activation, or normalization."""

    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        # A random down projection permits up to learn on the first update.
        # Zeroing both factors would leave both gradients zero forever.
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return features + self.up(self.down(features))


class HistoricalAdapter(nn.Module):
    """Shared control or independent head/tail adapters with matched starts."""

    def __init__(self, width, rank=8, mode="role"):
        super().__init__()
        if mode not in ("shared", "role"):
            raise ValueError("adapter mode must be shared or role")
        if not isinstance(rank, int) or not 1 <= rank <= width:
            raise ValueError("adapter rank must be between 1 and hidden width")
        self.width, self.rank, self.mode = width, rank, mode
        # Models are built on CPU before .to(device) in main.py. Only the new
        # module's initialization is isolated; training/evaluation RNG retain
        # the pinned author's shared stream. Subsequent diffusion construction
        # and backbone initialization therefore do not move to another seed.
        with torch.random.fork_rng(devices=[]):
            if mode == "shared":
                self.shared = ResidualAdapter(width, rank)
            else:
                self.head = ResidualAdapter(width, rank)
                # Equal starting projections, independent Parameters thereafter.
                self.tail = copy.deepcopy(self.head)

    def forward(self, head, tail):
        if (head.ndim != 2 or tail.shape != head.shape
                or head.shape[1] != self.width):
            raise ValueError("head and tail must both have shape [facts, hidden]")
        if head.device != tail.device or head.dtype != tail.dtype:
            raise ValueError("head/tail device and dtype must match")
        if not head.is_floating_point():
            raise ValueError("historical features must be floating point")
        if self.mode == "shared":
            return self.shared(head), self.shared(tail)
        return self.head(head), self.tail(tail)

    def extra_repr(self):
        return f"mode={self.mode}, width={self.width}, rank={self.rank}"
