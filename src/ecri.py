"""Evolution-Calibrated Replay Integration for chronological DGAR.

Statistics are data-only, computed before stage t from train[0:t]. No model
loss, held-out label, or current-stage triple enters the gate features.
"""
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvolutionConfig:
    window: int = 30
    min_support: int = 10
    sample_cap: int = 128
    permutations: int = 20
    kappa: float = 20.0
    seed: int = 1

    def __post_init__(self):
        if self.window < 1 or self.min_support < 1 or self.sample_cap < self.min_support:
            raise ValueError('ECRI requires window >= 1 and sample_cap >= min_support >= 1')
        if self.permutations < 1 or not math.isfinite(self.kappa) or self.kappa <= 0 or self.seed < 0:
            raise ValueError('ECRI requires positive permutations/kappa and a nonnegative seed')

    @classmethod
    def from_args(cls, args):
        return cls(args.ecri_window, args.ecri_min_support, args.ecri_sample_cap,
                   args.ecri_permutations, args.ecri_kappa, args.seed)


def role_js(left, right):
    """JSD in bits over directed relation roles; empty windows are unsupported."""
    a, b = Counter(role for role, _ in left), Counter(role for role, _ in right)
    result = 0.0
    for role in a.keys() | b.keys():
        p, q = a[role] / len(left), b[role] / len(right)
        mean = (p + q) / 2
        if p:
            result += .5 * p * math.log2(p / mean)
        if q:
            result += .5 * q * math.log2(q / mean)
    return max(0.0, min(1.0, result))


def neighbor_churn(left, right):
    a, b = set(left), set(right)
    return 1 - len(a & b) / len(a | b) if a or b else 0.0


class EvolutionHistory:
    """A bounded 2*window snapshot queue with a strict next-stage cursor.

    Inputs are Original's forward training triples, r in [0, R). A forward
    (s,r,o) contributes role r to s and role r+R to o, so inverse evidence is
    accounted for once. Passing already-expanded inverse triples is an error.
    """
    def __init__(self, num_entities, num_relations, config):
        if num_entities < 1 or num_relations < 1:
            raise ValueError('Positive entity and relation vocabulary sizes required')
        self.num_entities, self.num_relations = num_entities, num_relations
        self.config = config
        self.next_stage = 0
        self._snapshots = deque(maxlen=2 * config.window)
        self.summary = {}

    def _validate_triples(self, triples):
        triples = np.asarray(triples)
        if triples.size == 0:
            if triples.shape not in ((0,), (0, 3)):
                raise ValueError('Empty triples must have shape [0,3]')
            return np.empty((0, 3), dtype=np.int64)
        if triples.ndim != 2 or triples.shape[1] != 3 or triples.dtype.kind not in 'iu':
            raise ValueError('ECRI expects integer forward training triples [N,3]')
        if (triples[:, (0, 2)] < 0).any() or (triples[:, (0, 2)] >= self.num_entities).any():
            raise ValueError('ECRI entity ID outside the vocabulary')
        if (triples[:, 1] < 0).any() or (triples[:, 1] >= self.num_relations).any():
            raise ValueError('ECRI expects forward relation IDs; do not append inverse triples')
        return triples.astype(np.int64, copy=True)

    def observe(self, stage, train_triples):
        if stage != self.next_stage:
            raise ValueError(f'ECRI requires chronological stage {self.next_stage}, received {stage}')
        triples = self._validate_triples(train_triples)
        self._snapshots.append((stage, triples))
        self.next_stage += 1
        self.summary = {}

    def features(self, stage):
        if stage != self.next_stage:
            raise ValueError('ECRI features must precede observation of the current stage')
        groups = [defaultdict(list), defaultdict(list)]
        cutoff = stage - self.config.window
        for index, triples in self._snapshots:
            target = groups[int(index >= cutoff)]
            for subject, relation, object_ in triples.tolist():
                target[subject].append((relation, object_))
                target[object_].append((relation + self.num_relations, subject))
        values = np.zeros((self.num_entities, 3), dtype=np.float32)
        reliable = 0
        entities = sorted(groups[0].keys() & groups[1].keys()) if stage >= 2*self.config.window else []
        for entity in entities:
            old, recent = groups[0][entity], groups[1][entity]
            count = min(len(old), len(recent), self.config.sample_cap)
            if count < self.config.min_support:
                continue
            # Local RNG never consumes the model/HCP training random stream.
            rng = random.Random(self.config.seed + 1000003 * stage + 9176 * entity)
            old = rng.sample(old, count)
            recent = rng.sample(recent, count)
            observed_js, observed_churn = role_js(old, recent), neighbor_churn(old, recent)
            pooled = old + recent
            null_js = null_churn = 0.0
            for _ in range(self.config.permutations):
                rng.shuffle(pooled)
                a, b = pooled[:count], pooled[count:]
                null_js += role_js(a, b)
                null_churn += neighbor_churn(a, b)
            values[entity] = (
                max(0.0, observed_js - null_js / self.config.permutations),
                max(0.0, observed_churn - null_churn / self.config.permutations),
                count / (count + self.config.kappa))
            reliable += 1
        active = values[:, 2] > 0
        self.summary = {
            'snapshot': stage, 'prefix_last_snapshot': stage - 1,
            'reference_window': [max(0, stage - 2*self.config.window), max(0, cutoff)],
            'recent_window': [max(0, cutoff), stage], 'window_end_exclusive': True,
            'warmup': stage < 2*self.config.window,
            'reliable_entities': reliable, 'total_entities': self.num_entities,
            'role_excess_mean': float(values[active, 0].mean()) if reliable else 0.0,
            'churn_excess_mean': float(values[active, 1].mean()) if reliable else 0.0,
            'confidence_mean': float(values[active, 2].mean()) if reliable else 0.0,
        }
        return values

    def state_dict(self):
        return {'version': 1, 'num_entities': self.num_entities, 'num_relations': self.num_relations,
                'config': asdict(self.config), 'next_stage': self.next_stage,
                'snapshots': [(index, triples.copy()) for index, triples in self._snapshots]}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('Missing or unsupported ECRI history checkpoint')
        for key, expected in (('num_entities', self.num_entities), ('num_relations', self.num_relations),
                              ('config', asdict(self.config))):
            if state[key] != expected:
                raise ValueError(f'ECRI checkpoint {key} differs')
        next_stage = state['next_stage']
        if not isinstance(next_stage, int) or next_stage < 0:
            raise ValueError('Invalid ECRI checkpoint cursor')
        expected = list(range(max(0, next_stage - 2*self.config.window), next_stage))
        if [index for index, _ in state['snapshots']] != expected:
            raise ValueError('ECRI checkpoint history must be contiguous and strictly past')
        snapshots = [(index, self._validate_triples(triples)) for index, triples in state['snapshots']]
        self._snapshots = deque(snapshots, maxlen=2*self.config.window)
        self.next_stage = next_stage
        self.summary = {}


class EvolutionGate(nn.Module):
    """Confidence shrinks a monotone learned gate back to Original's alpha.

    Original alpha is unconstrained; it must not be sigmoid-transformed/clamped
    in the fallback. Positive slopes encode the explicit evolution hypothesis.
    Parameters use constant initialization, consuming no random numbers.
    """
    def __init__(self, bias=0.0, role_init=1.0, churn_init=1.0):
        super().__init__()
        if not all(math.isfinite(x) for x in (bias, role_init, churn_init)) or min(role_init, churn_init) <= 0:
            raise ValueError('Finite bias and positive finite ECRI slope initializers required')
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.role_raw = nn.Parameter(torch.tensor(role_init + math.log(-math.expm1(-role_init))))
        self.churn_raw = nn.Parameter(torch.tensor(churn_init + math.log(-math.expm1(-churn_init))))

    def forward(self, original_alpha, features):
        role, churn, confidence = features.unbind(dim=-1)
        adaptive = torch.sigmoid(self.bias + F.softplus(self.role_raw)*role
                                 + F.softplus(self.churn_raw)*churn)
        alpha = (1-confidence)*original_alpha + confidence*adaptive
        return alpha.unsqueeze(-1)


def record_stage(path, history, gate):
    # One small CPU transfer per stage for reporting, never per optimizer batch.
    weights = torch.stack((gate.bias.detach(), F.softplus(gate.role_raw.detach()),
                           F.softplus(gate.churn_raw.detach()))).cpu().tolist()
    record = {**history.summary, 'gate_bias': weights[0],
              'gate_role_weight': weights[1], 'gate_churn_weight': weights[2]}
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
