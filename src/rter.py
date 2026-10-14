"""Causal relation-transition evidence for DGAR entity scores.

Only observed training snapshots are accepted. Queries contain (subject,
relation), never their answer. This module has no learned/teacher state and
does not touch the experiment RNG. Times are DGAR snapshot indices.
"""
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
import random

import numpy as np
import torch

MODES = ('original', 'transition', 'repeat', 'popularity', 'shuffled')
PROTOCOL = 'rter-score-v1'


@dataclass
class Evidence:
    """Sparse bounded log-evidence; constant across model/noise realizations."""
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    shape: tuple
    statistics: dict
    popularity: np.ndarray = None

    @torch.no_grad()
    def apply(self, scores, beta):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('RTER beta must be finite and nonnegative')
        if tuple(scores.shape) != self.shape or not scores.is_floating_point():
            raise ValueError('RTER expects floating scores [queries, entities]')
        if beta == 0 or (not len(self.values) and self.popularity is None):
            return scores
        largest = float(np.max(self.popularity)) if self.popularity is not None else float(np.max(self.values))
        if beta > torch.finfo(scores.dtype).max / max(1., largest):
            raise ValueError('RTER residual coefficient overflows the score dtype')
        # One CPU->device transfer per evidence array; no GPU->CPU score copy.
        if self.popularity is not None:
            bias = torch.as_tensor(self.popularity, device=scores.device, dtype=scores.dtype)
            return scores + beta * bias.unsqueeze(0)
        result = scores.clone()
        rows = torch.as_tensor(self.rows, device=scores.device, dtype=torch.long)
        columns = torch.as_tensor(self.columns, device=scores.device, dtype=torch.long)
        values = torch.as_tensor(self.values, device=scores.device, dtype=scores.dtype)
        result[rows, columns] += beta * values
        return result


class RelationEvidence:
    """Incremental prefix index, with strictly increasing exclusive stop times.

    For each ordered entity pair, transitions connect consecutive *observed*
    snapshots. Equal-time relations form a set; its Cartesian transition receives
    total mass 1. Missing events are not negatives. Long gaps beyond window are
    ignored. Both transition counts and event evidence decay by exp(-gap/horizon).
    """
    def __init__(self, snapshots, entities, relations, horizon=30., smoothing=20.,
                 window=30, clip=3., channel='cross', shuffle_seed=1):
        if entities < 1 or relations < 1:
            raise ValueError('Positive entity/relation counts required')
        if any(not math.isfinite(x) or x <= 0 for x in (horizon, smoothing, clip)):
            raise ValueError('Positive finite horizon/smoothing/clip required')
        if not isinstance(window, Integral) or window < 1 or channel not in ('cross', 'all'):
            raise ValueError('Invalid transition window/channel')
        if not isinstance(shuffle_seed, Integral) or shuffle_seed < 0:
            raise ValueError('Invalid shuffle seed')
        self.snapshots, self.entities, self.relations = snapshots, entities, relations
        self.horizon, self.smoothing, self.window = horizon, smoothing, int(window)
        self.clip, self.channel, self.stop = clip, channel, 0
        self.counts = np.zeros((2 * relations, 2 * relations), dtype=np.float64)
        self.popularity = np.zeros(entities, dtype=np.float64)
        self.events = {}  # subject -> {(relation, candidate): (decayed mass, last time)}
        self.last_pair = {}  # (subject,candidate) -> (last observed time, relation set)
        self.count_time, self.transitions = 0, 0
        perm = list(range(relations))
        random.Random(int(shuffle_seed)).shuffle(perm)
        self.permutation = np.asarray(perm + [x + relations for x in perm])

    @classmethod
    def from_args(cls, args, snapshots, entities, relations):
        return cls(snapshots, entities, relations, args.rter_horizon,
                   args.rter_smoothing, args.rter_window, args.rter_clip,
                   args.rter_channel, args.rter_shuffle_seed)

    def _snapshot(self, stage):
        rows = np.asarray(self.snapshots[stage])
        if rows.size == 0:
            return set()
        if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in 'iu':
            raise ValueError('Training snapshot must be integer [facts,3]')
        if (rows[:, (0, 2)] < 0).any() or (rows[:, (0, 2)] >= self.entities).any():
            raise ValueError('Entity outside DGAR encoding')
        if (rows[:, 1] < 0).any() or (rows[:, 1] >= 2 * self.relations).any():
            raise ValueError('Relation outside DGAR encoding')
        # Normalize inverse caches/inputs, deduplicate within a time snapshot.
        facts = set()
        for s, r, o in rows.tolist():
            facts.add((s, r, o) if r < self.relations else (o, r-self.relations, s))
        return facts

    def advance(self, stop):
        if not isinstance(stop, Integral) or not self.stop <= stop <= len(self.snapshots):
            raise ValueError('Prefix must advance monotonically within the training timeline')
        for stage in range(self.stop, int(stop)):
            decay = math.exp(-(stage - self.count_time) / self.horizon)
            self.counts *= decay
            self.popularity *= decay
            self.count_time = stage
            pairs = {}
            for s, r, o in sorted(self._snapshot(stage)):
                self.popularity[s] += 1
                self.popularity[o] += 1
                for head, rel, tail in ((s, r, o), (o, r+self.relations, s)):
                    pairs.setdefault((head, tail), set()).add(rel)
                    bank = self.events.setdefault(head, {})
                    value, last = bank.get((rel, tail), (0., stage))
                    bank[rel, tail] = (value * math.exp(-(stage-last)/self.horizon)+1., stage)
            for pair, current in sorted(pairs.items()):
                previous = self.last_pair.get(pair)
                if previous is not None and stage-previous[0] <= self.window:
                    old = previous[1]
                    amount = 1. / (len(old) * len(current))
                    for r0 in sorted(old):
                        for r1 in sorted(current):
                            self.counts[r0, r1] += amount
                    self.transitions += 1
                self.last_pair[pair] = (stage, tuple(sorted(current)))
        # Arrays are always represented at exclusive query time stop.
        decay = math.exp(-(int(stop)-self.count_time)/self.horizon)
        self.counts *= decay
        self.popularity *= decay
        self.count_time, self.stop = int(stop), int(stop)
        return self

    def affinity(self, shuffled=False):
        affinity = np.zeros_like(self.counts)
        # Separate target direction marginals. Otherwise inverse augmentation
        # itself gives a trivial probability lift for every same-direction edge.
        for start in (0, self.relations):
            block = self.counts[:, start:start+self.relations]
            total = block.sum()
            if total == 0:
                continue
            marginal = (block.sum(axis=0)+1e-12)/(total+1e-12*self.relations)
            mass = block.sum(axis=1)
            conditional = (block+self.smoothing*marginal[None, :])/(mass[:, None]+self.smoothing)
            lift = np.maximum(np.log(conditional/marginal[None, :]), 0.)
            lift[mass == 0] = 0
            affinity[:, start:start+self.relations] = lift
        if shuffled:
            affinity = affinity[np.ix_(self.permutation, self.permutation)]
        if self.channel == 'cross':
            np.fill_diagonal(affinity, 0)
        return affinity

    def evidence(self, queries, mode='transition'):
        if mode not in MODES:
            raise ValueError('Unknown RTER mode')
        queries = np.asarray(queries)
        if queries.size == 0:
            queries = np.empty((0, 2), dtype=np.int64)
        if queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype.kind not in 'iu':
            raise ValueError('Only integer known (subject,relation) query pairs are accepted')
        if (queries[:, 0] < 0).any() or (queries[:, 0] >= self.entities).any():
            raise ValueError('Query entity outside encoding')
        if (queries[:, 1] < 0).any() or (queries[:, 1] >= 2*self.relations).any():
            raise ValueError('Query relation outside encoding')
        affinity = self.affinity(mode == 'shuffled') if mode in ('transition', 'shuffled') else None
        rows, columns, values, supported = [], [], [], 0
        cache = {}
        popularity = None
        if mode == 'popularity':
            popularity = np.minimum(np.log1p(self.popularity), self.clip)
        for i, (s, r) in enumerate(queries.tolist()):
            if mode in ('original', 'popularity'):
                continue
            if (s, r) not in cache:
                candidates = {}
                for (old_r, e), (mass, last) in self.events.get(s, {}).items():
                    weight = float(old_r == r) if mode == 'repeat' else affinity[old_r, r]
                    if weight > 0:
                        amount = mass * math.exp(-(self.stop-last)/self.horizon) * weight
                        if amount > 0:
                            candidates[e] = candidates.get(e, 0.)+amount
                cache[s, r] = [(e, min(math.log1p(value), self.clip))
                               for e, value in sorted(candidates.items())]
            items = cache[s, r]
            supported += bool(items)
            for e, value in items:
                rows.append(i)
                columns.append(e)
                values.append(value)
        if popularity is not None:
            supported = len(queries) if np.any(popularity) else 0
        stats = dict(stage=self.stop, history_stop=self.stop, mode=mode,
                     channel=self.channel, queries=len(queries), supported_queries=supported,
                     boosted_entries=len(values) if popularity is None else int(np.count_nonzero(popularity))*len(queries),
                     transitions_observed=self.transitions, transition_mass=float(self.counts.sum()),
                     history_edges=sum(len(x) for x in self.events.values()),
                     mean_unit_residual=float(np.mean(popularity[popularity>0])) if popularity is not None and np.any(popularity)
                     else float(np.mean(values)) if values else 0.)
        return Evidence(np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64),
                        np.asarray(values, dtype=np.float64), (len(queries), self.entities), stats, popularity)


def known_queries(forward_triples, relations):
    """DGAR forward and inverse query order; returns no target column."""
    triples = np.asarray(forward_triples)
    forward = triples[:, :2].copy()
    inverse = triples[:, (2, 1)].copy()
    inverse[:, 1] = np.where(inverse[:, 1] < relations,
                            inverse[:, 1]+relations, inverse[:, 1]-relations)
    return np.concatenate((forward, inverse))


def record(path, evidence, beta, phase):
    row = dict(protocol=PROTOCOL, phase=phase, beta=beta, **evidence.statistics)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True)+'\n')
