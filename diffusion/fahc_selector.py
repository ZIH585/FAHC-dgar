"""Forgetting-aware selection of historical context prompts (HCPs).

The snapshot files produced by :func:`rgcn.utils.get_history` map a subject
entity ID (as a string) to a set of ``(subject, relation, object)`` triples.
This module deliberately does not depend on the model: the caller supplies
per-triple old/probe losses, then uses their positive increases as risks.

Coverage uses a concave harmonic reward. For each feature value x (relation,
entity, or time bin), if x has appeared in n selected HCPs, the next HCP
containing x earns 1/(n+1). Each feature family's sum is divided by the number
of distinct values in the candidate pool, so eta has a stable interpretation
across candidate pool sizes. Entity coverage can be disabled with
``entity_weight=0`` when reproducing the relation/time-only objective.
"""

from __future__ import annotations

import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple


Triple = Tuple[int, int, int]


@dataclass(frozen=True)
class HCPCandidate:
    time_index: int
    triples: Tuple[Triple, ...]


@dataclass(frozen=True)
class SelectionResult:
    indices: Tuple[int, ...]
    triples: Tuple[Triple, ...]
    selected: Tuple[HCPCandidate, ...]
    score_by_index: Mapping[int, float]

    @property
    def nonempty_count(self) -> int:
        """Number of selected time slices with at least one relevant fact."""
        return sum(bool(candidate.triples) for candidate in self.selected)

    @property
    def fact_count(self) -> int:
        """Actual number of unique facts after applying the optional cap."""
        return len(self.triples)


def sample_scoring_candidates(
    candidates: Sequence[HCPCandidate], k: int, fact_budget: int, seed: int
) -> list[HCPCandidate]:
    """Deprecated approximation: sample a replay quota per candidate.

    This changes the proposed risk and coverage objective because selection
    only sees sampled facts. FAHC training scores complete HCP candidates and
    applies the separate replay cap *after* time-slice selection. Retained for
    explicit scoring-cost experiments and older callers.
    """
    if k < 1 or fact_budget < k:
        raise ValueError("a controlled HCP run needs k >= 1 and fact_budget >= k")
    if not candidates:
        return []
    quota = fact_budget // min(k, len(candidates))
    sampled = []
    for candidate in candidates:
        if len(candidate.triples) <= quota:
            triples = candidate.triples
        else:
            sampler = random.Random(f"{seed}:{candidate.time_index}:score")
            triples = tuple(sorted(sampler.sample(candidate.triples, quota)))
        sampled.append(HCPCandidate(candidate.time_index, triples))
    return sampled


def materialize_replay(
    selection: SelectionResult, full_candidates: Sequence[HCPCandidate],
    fact_budget: int, seed: int
) -> SelectionResult:
    """Cap the unique fact union of selected full HCPs after selection.

    Each selected time slice receives an independently seeded fact shuffle.
    A seeded shuffle of slice order resolves the final incomplete round. The
    selection risks and coverage remain based on complete HCPs; the replay
    budget does not silently alter the time-slice objective.
    """
    if fact_budget < 0:
        raise ValueError("fact_budget must be nonnegative")
    by_index = {candidate.time_index: candidate for candidate in full_candidates}
    if len(by_index) != len(full_candidates):
        raise ValueError("candidate time indices must be distinct")
    if any(index not in by_index for index in selection.indices):
        raise ValueError("selected HCP is absent from full candidates")
    selected_full = [by_index[index] for index in selection.indices]
    replay = _budgeted_union(selected_full, fact_budget, seed)
    return SelectionResult(selection.indices, replay,
                           selection.selected, selection.score_by_index)


def eligible_history_indices(current_time: int) -> list[int]:
    """Match DGAR's normal training HCP range: snapshots 0 through t-2."""
    if current_time < 0:
        raise ValueError("current_time must be nonnegative")
    return list(range(max(0, current_time - 1)))


def recent_candidate_indices(current_time: int, k: int) -> list[int]:
    """Return the k most recent indices in DGAR's eligible history range."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    return eligible_history_indices(current_time)[-k:] if k else []


def sample_candidate_indices(
    current_time: int,
    k: int,
    multiplier: float = 2.0,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """Draw up to ceil(multiplier*k) distinct HCP snapshots from DGAR's range."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("multiplier must be finite and at least 1")
    eligible = eligible_history_indices(current_time)
    count = min(len(eligible), math.ceil(multiplier * k))
    if count == len(eligible):
        return eligible
    sampler = rng if rng is not None else random
    return sorted(sampler.sample(eligible, count))


def load_hcp_candidates(
    dataset: str,
    indices: Iterable[int],
    query_subjects: Iterable[int],
    history_dir: Optional[Path] = None,
) -> list[HCPCandidate]:
    """Read existing HCP files and retain facts matching current subjects.

    Each snapshot is read once, and triples are sorted for repeatable scoring,
    tie breaking, and fact-budget truncation. Empty HCPs remain candidates so
    random and recent baselines preserve the same time-index semantics.
    """
    directory = (
        Path(history_dir)
        if history_dir is not None
        else Path(__file__).resolve().parents[1] / "data" / dataset / "history_snap_v2"
    )
    subjects = {str(int(subject)) for subject in query_subjects}
    candidates = []
    seen = set()
    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            raise ValueError("historical snapshot indices must be nonnegative")
        if index in seen:
            raise ValueError("historical snapshot indices must be distinct")
        seen.add(index)
        with (directory / f"snap_{index}.pkl").open("rb") as stream:
            history = pickle.load(stream)
        facts = set()
        for subject in subjects:
            for triple in history.get(subject, ()):
                if len(triple) != 3:
                    raise ValueError(f"malformed triple in historical snapshot {index}")
                facts.add(tuple(int(value) for value in triple))
        candidates.append(HCPCandidate(index, tuple(sorted(facts))))
    return candidates


def positive_loss_increase(
    old_losses: Sequence[float], probe_losses: Sequence[float]
) -> float:
    """Return mean_q max(loss_probe(q) - loss_old(q), 0).

    The two sequences must be aligned to the same ordered triples. Empty HCPs
    have risk zero because there are no facts to replay or score.
    """
    if len(old_losses) != len(probe_losses):
        raise ValueError("old and probe losses must have the same length")
    if len(old_losses) == 0:
        return 0.0
    total = 0.0
    for old, probe in zip(old_losses, probe_losses):
        old_value, probe_value = float(old), float(probe)
        if not math.isfinite(old_value) or not math.isfinite(probe_value):
            raise ValueError("old and probe losses must be finite")
        total += max(probe_value - old_value, 0.0)
    return total / len(old_losses)


def _budgeted_union(
    selected: Sequence[HCPCandidate], fact_budget: Optional[int], seed: int = 0
) -> Tuple[Triple, ...]:
    if fact_budget is None:
        return tuple(sorted({triple for candidate in selected for triple in candidate.triples}))
    if fact_budget < 0:
        raise ValueError("fact_budget must be nonnegative")
    if fact_budget == 0:
        return ()
    # Shuffle within each slice and randomize which slices receive the extra
    # fact in a partial round. Duplicate triples across slices count once.
    selected = list(selected)
    random.Random(f"{seed}:slice-order").shuffle(selected)
    streams = []
    for candidate in selected:
        facts = list(candidate.triples)
        random.Random(f"{seed}:{candidate.time_index}:fill").shuffle(facts)
        streams.append(iter(facts))
    result = []
    seen = set()
    while len(result) < fact_budget:
        progressed = False
        for stream in streams:
            for triple in stream:
                if triple not in seen:
                    seen.add(triple)
                    result.append(triple)
                    progressed = True
                    break
            if len(result) == fact_budget:
                break
        if not progressed:
            break
    return tuple(result)


def select_hcp(
    candidates: Sequence[HCPCandidate],
    k: int,
    risks: Optional[Mapping[int, float]] = None,
    mode: str = "risk+coverage",
    eta: float = 1.0,
    time_bin_size: int = 10,
    fact_budget: Optional[int] = None,
    rng: Optional[random.Random] = None,
    entity_weight: float = 1.0,
    relation_modulus: Optional[int] = None,
) -> SelectionResult:
    """Select exactly min(k, number of candidates) historical time slices.

    Modes are ``risk+coverage``, ``risk``, ``coverage``, ``random``, and
    ``recent``. The greedy modes maximize the marginal gain of

      sum(P in M) risk(P) + eta * (C_relation + C_time +
                                    entity_weight * C_entity),

    with irrelevant terms set to zero by the chosen mode. ``C`` is normalized
    harmonic feature coverage described in the module docstring. ``random``
    samples without replacement; ``recent`` selects largest time indices.
    Greedy modes choose nonempty HCPs first, filling remaining slots with
    empty historical slices if fewer than k have relevant facts. ``random``
    and ``recent`` retain DGAR's time-slice behavior, including empty slices.
    ``fact_budget`` caps the unique union after selecting time slices. It must
    also be used by the corresponding baseline to compare equal budgets.
    """
    if mode == "risk_coverage":
        mode = "risk+coverage"
    if mode not in {"risk+coverage", "risk", "coverage", "random", "recent"}:
        raise ValueError(f"unknown HCP selection mode: {mode}")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if time_bin_size <= 0:
        raise ValueError("time_bin_size must be positive")
    if fact_budget is not None and fact_budget < 0:
        raise ValueError("fact_budget must be nonnegative")
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if not math.isfinite(entity_weight) or entity_weight < 0:
        raise ValueError("entity_weight must be finite and nonnegative")
    if relation_modulus is not None and relation_modulus <= 0:
        raise ValueError("relation_modulus must be positive")

    by_index = {candidate.time_index: candidate for candidate in candidates}
    if len(by_index) != len(candidates):
        raise ValueError("candidate time indices must be distinct")
    ordered = sorted(candidates, key=lambda candidate: candidate.time_index)
    count = min(k, len(ordered))
    if count == 0:
        return SelectionResult((), (), (), {})

    if mode == "random":
        sampler = rng if rng is not None else random
        selected = sampler.sample(ordered, count)
        scores = {candidate.time_index: 0.0 for candidate in selected}
    elif mode == "recent":
        selected = list(reversed(ordered[-count:]))
        scores = {candidate.time_index: 0.0 for candidate in selected}
    else:
        use_risk = mode in {"risk+coverage", "risk"}
        use_coverage = mode in {"risk+coverage", "coverage"}
        tie_rng = rng if rng is not None else random.Random(0)
        if use_risk and risks is None:
            raise ValueError("risk-based selection requires per-candidate risks")
        risk_values = {}
        for candidate in ordered:
            value = (
                float(risks[candidate.time_index])
                if use_risk and candidate.triples else 0.0
            )
            if not math.isfinite(value) or value < 0:
                raise ValueError("candidate risks must be finite and nonnegative")
            risk_values[candidate.time_index] = value

        relations = {candidate.time_index: {
            triple[1] % relation_modulus if relation_modulus else triple[1]
            for triple in candidate.triples}
                     for candidate in ordered}
        entities = {candidate.time_index: {entity for s, _, o in candidate.triples
                                          for entity in (s, o)} for candidate in ordered}
        bins = {candidate.time_index: {candidate.time_index // time_bin_size}
                if candidate.triples else set() for candidate in ordered}
        universe_sizes = (
            max(1, len(set().union(*relations.values()))),
            max(1, len(set().union(*bins.values()))),
            max(1, len(set().union(*entities.values()))),
        )
        relation_counts, time_counts, entity_counts = Counter(), Counter(), Counter()
        remaining = list(ordered)
        selected = []
        scores = {}
        for _ in range(count):
            best_score = -math.inf
            tied = []
            # Empty HCPs carry no replay facts, so fill with them only after
            # exhausting nonempty candidates. This still returns k indices.
            active = [candidate for candidate in remaining if candidate.triples]
            for candidate in active if active else remaining:
                index = candidate.time_index
                coverage_gain = 0.0
                if use_coverage:
                    coverage_gain = (
                        sum(1 / (relation_counts[x] + 1) for x in relations[index]) / universe_sizes[0]
                        + sum(1 / (time_counts[x] + 1) for x in bins[index]) / universe_sizes[1]
                        + entity_weight * sum(1 / (entity_counts[x] + 1) for x in entities[index]) / universe_sizes[2]
                    )
                score = risk_values[index] + eta * coverage_gain
                if score > best_score:
                    best_score = score
                    tied = [candidate]
                elif score == best_score:
                    tied.append(candidate)
            best = tied[0] if len(tied) == 1 else tie_rng.choice(tied)
            selected.append(best)
            scores[best.time_index] = best_score
            remaining.remove(best)
            relation_counts.update(relations[best.time_index])
            time_counts.update(bins[best.time_index])
            entity_counts.update(entities[best.time_index])

    chosen = tuple(selected)
    return SelectionResult(
        tuple(candidate.time_index for candidate in chosen),
        _budgeted_union(chosen, fact_budget),
        chosen,
        scores,
    )
