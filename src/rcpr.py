"""Relation-conditioned prototype replay. All contexts come from Original HCP.

A context label is r at a head occurrence and inverse(r) at a tail occurrence.
No labels from the queries being scored are used to construct the score residual.
Only integer layouts are cached; differentiable features live for one forward.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = ('original', 'mean-shared', 'mean-head', 'prototype', 'shuffled')
HEAD_MODES = frozenset(('mean-head', 'prototype', 'shuffled'))


class SharedAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if not 1 <= rank <= width:
            raise ValueError('adapter rank must be in [1,width]')
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(width, rank, bias=False)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


@dataclass
class Layout:
    ids: torch.Tensor
    contexts: torch.Tensor
    pair_entities: torch.Tensor
    pair_contexts: torch.Tensor
    pair_inverse: torch.Tensor
    pair_counts: torch.Tensor
    num_entities: int
    num_relations: int


@dataclass
class Context:
    layout: Layout
    residuals: torch.Tensor
    means: torch.Tensor

    def detached(self):
        return Context(self.layout, self.residuals.detach(), self.means.detach())


def make_layout(tags, num_entities, num_relations, shuffled=False, seed=0):
    if tags.ndim != 2 or tags.shape[1] != 3 or tags.dtype != torch.long:
        raise ValueError('HCP tags must be int64 [facts,3]')
    if num_entities < 1 or num_relations < 1 or seed < 0:
        raise ValueError('invalid entity/relation counts or shuffle seed')
    # Once per new bank (cached by the owning model), never per scoring chunk.
    if len(tags) and (bool((tags[:, (0, 2)] < 0).any())
                     or bool((tags[:, (0, 2)] >= num_entities).any())
                     or bool((tags[:, 1] < 0).any())
                     or bool((tags[:, 1] >= 2 * num_relations).any())):
        raise ValueError('entity or directed relation ID out of range')
    ids = torch.cat((tags[:, 0], tags[:, 2]))
    rel = tags[:, 1]
    contexts = torch.cat((rel, (rel + num_relations) % (2 * num_relations)))
    if shuffled and len(tags):
        # Permute relation labels within each entity AND original head/tail role.
        # This keeps all entity-role-context counts and global means unchanged.
        roles = torch.cat((torch.zeros_like(rel), torch.ones_like(rel)))
        groups = 2 * ids + roles
        ordered = torch.argsort(groups, stable=True)
        generator = torch.Generator(device=tags.device).manual_seed(seed)
        random_order = torch.randperm(len(ids), device=tags.device, generator=generator)
        permuted = random_order[torch.argsort(groups[random_order], stable=True)]
        contexts = contexts.index_copy(0, ordered, contexts[permuted])
    keys, inverse, counts = torch.unique(ids * (2 * num_relations) + contexts,
                                          sorted=True, return_inverse=True, return_counts=True)
    return Layout(ids, contexts, keys // (2 * num_relations), keys % (2 * num_relations),
                  inverse, counts, num_entities, num_relations)


def entity_mean(layout, occurrences):
    bank = occurrences.new_zeros(layout.num_entities, occurrences.shape[1])
    bank = bank.index_add(0, layout.ids, occurrences)
    counts = torch.bincount(layout.ids, minlength=layout.num_entities).to(occurrences.dtype)
    return bank / counts.clamp_min(1).unsqueeze(1)


def make_context(layout, occurrences, mean_bank=None):
    if occurrences.ndim != 2 or len(occurrences) != len(layout.ids):
        raise ValueError('occurrences must have shape [2*facts,hidden]')
    if occurrences.device != layout.ids.device:
        raise ValueError('HCP metadata and features must be on the same device')
    if mean_bank is None:
        mean_bank = entity_mean(layout, occurrences)
    if mean_bank.shape != (layout.num_entities, occurrences.shape[1]):
        raise ValueError('entity mean bank has wrong shape')
    prototypes = occurrences.new_zeros(len(layout.pair_counts), occurrences.shape[1])
    prototypes = prototypes.index_add(0, layout.pair_inverse, occurrences)
    prototypes = prototypes / layout.pair_counts.to(occurrences.dtype).unsqueeze(1)
    means = mean_bank[layout.pair_entities]
    return Context(layout, prototypes - means, means)


class ScoreResidual(nn.Module):
    """s+=<Wq query,Wp delta>/sqrt(k), with a zero initial output.

    Work on sparse present entity-context pairs, rather than allocating an
    [entities,relations,hidden] bank. Chunk only the additive score correction;
    the original decoder's BN/dropout batch and number of calls are unchanged.
    """
    def __init__(self, width, rank, chunk_size=256):
        super().__init__()
        if not 1 <= rank <= width or chunk_size < 1:
            raise ValueError('invalid residual rank or chunk size')
        with torch.random.fork_rng(devices=[]):
            self.query = nn.Linear(width, rank, bias=False)
            self.prototype = nn.Linear(width, rank, bias=False)
            nn.init.zeros_(self.query.weight)
        self.rank, self.chunk_size = rank, chunk_size

    def forward(self, scores, query, query_keys, context, mean_control=False):
        if query_keys.shape != (len(query), 2) or query_keys.dtype != torch.long:
            raise ValueError('score residual accepts [subject,relation] keys only')
        if scores.shape != (len(query), context.layout.num_entities):
            raise ValueError('score shape differs from the full candidate bank')
        layout = context.layout
        if len(query) == 0 or len(layout.pair_entities) == 0:
            return scores
        values = context.means if mean_control else context.residuals
        projected = self.prototype(values)
        queries = self.query(query) / math.sqrt(self.rank)
        inverse = (query_keys[:, 1] + layout.num_relations) % (2 * layout.num_relations)
        pieces = []
        for start in range(0, len(query), self.chunk_size):
            end = min(start + self.chunk_size, len(query))
            matches = inverse[start:end, None] == layout.pair_contexts[None, :]
            differences = (queries[start:end] @ projected.T) * matches
            correction = scores.new_zeros(end - start, scores.shape[1])
            correction = correction.scatter_add(1, layout.pair_entities.expand(end - start, -1), differences)
            pieces.append(scores[start:end] + correction)
        return torch.cat(pieces)


@torch.no_grad()
def context_statistics(context, occurrences):
    layout = context.layout
    counts = torch.bincount(layout.ids, minlength=layout.num_entities).to(occurrences.dtype)
    bank = entity_mean(layout, occurrences)
    total = (occurrences - bank[layout.ids]).square().sum()
    explained = (context.residuals.square().sum(1) * layout.pair_counts).sum()
    fraction = explained / total.clamp_min(torch.finfo(occurrences.dtype).tiny)
    return {'facts': len(layout.ids) // 2, 'entities': int((counts > 0).sum()),
            'entity_relation_pairs': len(layout.pair_entities),
            'relation_explained_scatter': float(fraction.cpu()),
            'total_within_entity_scatter': float(total.cpu())}
