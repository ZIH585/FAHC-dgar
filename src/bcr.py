"""Direction-separated, training-only historical context retrieval for DGAR.

No old model, new loss, or learned state. Indexes are derived lazily from the
original training snapshots. The author cache is never modified.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import random

import torch


PROTOCOL = 'bcr-directional-v1'


def enabled(args):
    return getattr(args, 'bcr_mode', 'original') != 'original'


@dataclass
class Replay:
    tags: torch.Tensor
    features: object

    @property
    def objects(self):
        return None if self.features is None else self.features[:, -1, :]


class HistoricalIndex:
    """Selected past training snapshots only; forward facts remain forward."""

    def __init__(self, train, num_relations, num_entities):
        self.train = train
        self.num_relations = num_relations
        self.num_entities = num_entities
        self.indexes = {}

    def snapshot(self, index, stage, stop):
        if not 0 <= index < stop <= stage <= len(self.train):
            raise ValueError('BCR may only retrieve selected, strictly past training snapshots')
        if index not in self.indexes:
            outgoing, incoming, facts = {}, {}, set()
            for row in self.train[index]:
                s, r, o = map(int, row)
                if not (0 <= s < self.num_entities and 0 <= o < self.num_entities
                        and 0 <= r < self.num_relations):
                    raise ValueError('BCR indexes forward training triples, not augmented inverse triples')
                fact = (s, r, o)
                facts.add(fact)
                outgoing.setdefault(s, set()).add(fact)
                incoming.setdefault(o, set()).add(fact)
            self.indexes[index] = (outgoing, incoming, facts)
        return self.indexes[index]

    def pools(self, subjects, times, stage, stop):
        subjects = {int(s) for s in subjects}
        if any(not 0 <= s < self.num_entities for s in subjects):
            raise ValueError('Invalid known query entity')
        outgoing, incident, all_facts = set(), set(), set()
        for index in times:
            out, inc, facts = self.snapshot(index, stage, stop)
            all_facts.update(facts)
            for subject in subjects:
                outgoing.update(out.get(subject, ()))
                incident.update(out.get(subject, ()))
                incident.update(inc.get(subject, ()))
        return outgoing, incident, all_facts


def choose(pool, budget, seed):
    """Independent selection RNG; canonical triples, deterministic row order."""
    if budget < 0:
        raise ValueError('Negative BCR budget')
    ordered = sorted(pool)
    if len(ordered) > budget:
        ordered = random.Random(seed).sample(ordered, budget)
    return sorted(ordered)


@contextmanager
def isolated_generation(generator, guider, device, seed):
    """Prevent one direction's entities/noise from altering the other's buffers.

    Keep original TRAIN/EVAL modes. Preserve running buffers and private torch
    randomness, while the caller consumes one shared seed draw per evaluation.
    """
    models = [generator] if guider is generator else [generator, guider]
    buffers = [(buffer, buffer.detach().clone()) for model in models
               for buffer in model.buffers()]
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)


class BCR:
    def __init__(self, args, train, num_relations, num_entities):
        self.args = args
        self.index = HistoricalIndex(train, num_relations, num_entities)
        self.cache_dir = Path(args.data_root) / args.dataset / 'history_snap_v2'

    def times(self, stage, final=False):
        # Exactly the author's different train and final-evaluation ranges.
        if not final and stage < 2:
            return [], 0
        stop = stage if final else stage - 1
        threshold = stage - 1 if final else stage - 2
        if threshold < self.args.reply_batch:
            return list(range(stop)), stop
        return random.sample(list(range(stop)), self.args.reply_batch), stop

    def legacy_budget(self, subjects, times):
        """Original cache retrieval for these KNOWN heads, normalized to forward.

        Supports author caches containing optional inverse records without
        inserting inverse IDs into the diffusion generator or double-counting.
        """
        facts = set()
        for index in times:
            with (self.cache_dir / f'snap_{index}.pkl').open('rb') as stream:
                cache = pickle.load(stream)
            for s in set(map(int, subjects)):
                for h, r, t in cache.get(str(s), ()):
                    h, r, t = int(h), int(r), int(t)
                    if r >= self.index.num_relations:
                        h, r, t = t, r - self.index.num_relations, h
                    facts.add((h, r, t))
        return len(facts)

    def retrieve(self, subjects, times, stage, stop, seed, budget=None):
        out, incident, all_facts = self.index.pools(subjects, times, stage, stop)
        # Direction-specific outgoing capacity; every control uses this same
        # budget. Training explicitly supplies Original's forward-head budget.
        if budget is None:
            budget = len(out)
        pools = {'outgoing': out, 'incident': incident, 'random': all_facts}
        pool = pools[self.args.bcr_mode]
        selected = choose(pool, budget, seed)
        stats = dict(budget=budget, selected=len(selected), outgoing=len(out),
                     incident=len(incident), global_pool=len(all_facts),
                     incoming_only=len(incident - out),
                     selected_incoming_only=len(set(selected) & (incident - out)),
                     selected_unrelated=len(set(selected) - incident),
                     selection_seed=seed,
                     known_entities=len(set(map(int, subjects))))
        return selected, stats

    def record(self, phase, stage, times, stats):
        if self.args.run_dir is None:
            return
        row = dict(protocol=PROTOCOL, phase=phase, stage=stage,
                   mode=self.args.bcr_mode, times=list(times), contexts=stats)
        with (Path(self.args.run_dir) / 'bcr.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')

    @staticmethod
    def generate(args, rows, generator, guider, history, last, static, stage, device, cuda):
        tags = torch.as_tensor(rows, dtype=torch.long, device=device).reshape(-1, 3)
        features = None
        if len(tags):
            with torch.no_grad():
                result = generator(None, tags, args, False, cuda,
                    static_graph=static, ct=stage, model=guider,
                    history_glist=history, triples=last)
                features = result[1].detach()
            if features.shape != (len(tags), 3, args.n_hidden):
                raise ValueError('Unexpected generated BCR feature shape')
        return Replay(tags, features)

    def training(self, stage, triples, generator, guider, history, last, static, cuda):
        times, stop = self.times(stage)
        if not times:
            self.record('train', stage, times, [])
            return triples, None, None
        # Both endpoints are OBSERVED training facts, so both are legitimate.
        heads = self.index.train[stage][:, 0].tolist()
        subjects = set(heads) | set(self.index.train[stage][:, 2].tolist())
        budget = self.legacy_budget(heads, times)
        seed = random.getrandbits(63)
        rows, stats = self.retrieve(subjects, times, stage, stop, seed, budget)
        replay = self.generate(self.args, rows, generator, guider, history,
                               last, static, stage, triples.device, cuda)
        self.record('train', stage, times, [stats])
        # The existing main/get_loss/diffusion loop still receives one tensor.
        return triples, replay.features, replay.tags if len(replay.tags) else None

    def evaluation(self, stage, forward_heads, inverse_heads, generator, guider,
                   history, last, static, device, cuda, final=False,
                   times=None, stop=None, seed=None, phase='validation'):
        """Receives only known heads; there is no target/relation-label argument.

        Each direction's retrieval, budget, sampling and generation are
        independent of the opposite direction's known entities.
        """
        if times is None:
            times, stop = self.times(stage, final)
        # One fixed-size shared RNG draw, independent of either label/length.
        if seed is None:
            seed = random.getrandbits(63)
        replays, stats = [], []
        for direction, heads in enumerate((forward_heads, inverse_heads)):
            branch_seed = (seed + 104729 * direction) % (2**63 - 1)
            rows, numbers = self.retrieve(heads, times, stage, stop, branch_seed)
            with isolated_generation(generator, guider, device, branch_seed):
                replays.append(self.generate(self.args, rows, generator, guider,
                    history, last, static, stage, device, cuda))
            stats.append(numbers)
        self.record(phase, stage, times, stats)
        return tuple(replays)


@torch.no_grad()
def predict(model, history, num_relations, static, triples, replays, cuda):
    """Keep original full 2B decoder batches, then take each direction's rows.

    ConvTransE has a singleton special case; do not score queries one by one.
    model is EVAL, so opposite-direction query inputs do not alter BN buffers.
    """
    n = len(triples)
    if not n:
        device = triples.device
        return (triples.reshape(0, 3), torch.empty(0, model.num_ents, device=device),
                torch.empty(0, 2 * num_relations, device=device))
    if model.training:
        raise ValueError('BCR directional prediction requires model.eval()')
    if replays is None or len(replays) != 2:
        raise ValueError('BCR requires separate forward and inverse replay contexts')
    results = []
    for direction, replay in enumerate(replays):
        expanded, scores, relation_scores = model.predict(history, num_relations,
            static, triples, replay.features, replay.objects, replay.tags, cuda)
        sl = slice(direction * n, (direction + 1) * n)
        results.append((expanded[sl], scores[sl], relation_scores[sl]))
    return tuple(torch.cat((results[0][i], results[1][i])) for i in range(3))
