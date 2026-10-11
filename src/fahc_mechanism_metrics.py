"""Standard-library statistics for the FAHC risk mechanism diagnostic.

Candidate-level metrics weight each HCP equally, matching the selection score.
They measure ranking stability and prediction of observed HCP loss increases;
they do not establish an improvement in held-out link-prediction performance.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from numbers import Integral, Real
import random
from typing import Mapping, Sequence


def _validated_vector(name: str, values: Mapping[int, float],
                      candidate_ids: Sequence[int]) -> list[float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping keyed by candidate ID")
    if set(values) != set(candidate_ids):
        raise ValueError(f"{name} must contain exactly the candidate IDs")
    result = []
    for candidate_id in candidate_ids:
        value = values[candidate_id]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{candidate_id}] must be finite and nonnegative")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name}[{candidate_id}] must be finite and nonnegative") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name}[{candidate_id}] must be finite and nonnegative")
        result.append(number)
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return ascending one-based ranks with average ranks for exact ties."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[ordered[position]] = average
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    center = (len(left) + 1) / 2.0
    left_centered = [value - center for value in left_ranks]
    right_centered = [value - center for value in right_ranks]
    left_variance = math.fsum(value * value for value in left_centered)
    right_variance = math.fsum(value * value for value in right_centered)
    if left_variance == 0 or right_variance == 0:
        return None
    covariance = math.fsum(a * b for a, b in zip(left_centered, right_centered))
    correlation = covariance / math.sqrt(left_variance * right_variance)
    return max(-1.0, min(1.0, correlation))


def _topk(candidate_ids: Sequence[int], values: Sequence[float], k: int,
          seed: int, stream: str) -> list[int]:
    """Randomize exact ties using a separate reproducible stream per ranking."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    # IDs are canonicalized before this call. Input ordering cannot change ties.
    decorated = [(value, rng.random(), candidate_id)
                 for candidate_id, value in zip(candidate_ids, values)]
    decorated.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    return [entry[2] for entry in decorated[:k]]


def _vector_stats(values: Sequence[float]) -> dict[str, int | float | None]:
    counts = Counter(values)
    count = len(values)
    zeros = counts.get(0.0, 0)
    tied_items = sum(size for size in counts.values() if size > 1)
    return {
        "count": count,
        "unique_count": len(counts),
        "zero_count": zeros,
        "zero_fraction": zeros / count if count else None,
        "tied_item_count": tied_items,
        "tied_item_fraction": tied_items / count if count else None,
        "tied_group_count": sum(size > 1 for size in counts.values()),
    }


def _mean_nonnegative(values: Sequence[float]) -> float | None:
    if not values:
        return None
    scale = max(values)
    if scale == 0:
        return 0.0
    # Normalization avoids overflowing a sum of individually finite inputs.
    return (math.fsum(value / scale for value in values) / len(values)) * scale


def summarize_mechanism(candidate_ids: Sequence[int],
                        risk_a: Mapping[int, float],
                        risk_b: Mapping[int, float],
                        observed: Mapping[int, float],
                        k: int, seed: int = 0) -> dict:
    """Summarize two independent probes and a common observed loss increase.

    Inputs must contain the same unique nonnegative integer candidate IDs and
    finite nonnegative scores. Empty HCPs must already have been removed by the
    caller. ``observed`` should use the same per-fact positive-increase formula
    as probe risk. A constant vector has undefined Spearman correlation, which
    is reported as ``None`` rather than implying stability or predictive power.

    ``effective_k = min(k, n_candidates)``. The random expectation for top-k
    overlap fraction is ``effective_k / n_candidates``. When every candidate
    is selected, overlap is necessarily one and conveys no selection evidence.
    Exact ties are randomized independently for A, B, their mean, and the oracle
    with recorded, reproducible streams. An all-zero vector therefore cannot
    produce a systematically perfect overlap through a shared tie-break rule.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ids = list(candidate_ids)
    if any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0
           for item in ids):
        raise ValueError("candidate IDs must be nonnegative integers")
    ids = [int(item) for item in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    ids.sort()
    a = _validated_vector("risk_a", risk_a, ids)
    b = _validated_vector("risk_b", risk_b, ids)
    actual = _validated_vector("observed", observed, ids)
    # This form avoids both overflow at large scores and halving two equal
    # subnormal scores to zero before adding them back together.
    average = [min(left, right) + (max(left, right) - min(left, right)) / 2.0
               for left, right in zip(a, b)]
    n = len(ids)
    effective_k = min(int(k), n)
    rankings = {
        "a": _topk(ids, a, effective_k, int(seed), "risk_a"),
        "b": _topk(ids, b, effective_k, int(seed), "risk_b"),
        "mean": _topk(ids, average, effective_k, int(seed), "risk_mean"),
        "observed": _topk(ids, actual, effective_k, int(seed), "observed"),
    }
    a_set, b_set = set(rankings["a"]), set(rankings["b"])
    observed_by_id = dict(zip(ids, actual))
    selected_actual = [observed_by_id[item] for item in rankings["mean"]]
    oracle_actual = [observed_by_id[item] for item in rankings["observed"]]
    all_mean = _mean_nonnegative(actual)
    selected_mean = _mean_nonnegative(selected_actual)
    observed_scale = max(actual, default=0.0)
    oracle_capture = None
    if observed_scale > 0:
        oracle_sum = math.fsum(value / observed_scale for value in oracle_actual)
        selected_sum = math.fsum(value / observed_scale for value in selected_actual)
        oracle_capture = min(1.0, selected_sum / oracle_sum)

    return {
        "n_candidates": n,
        "requested_k": int(k),
        "effective_k": effective_k,
        "selection_has_choice": n > effective_k,
        "tie_seed": int(seed),
        "risk_mean": dict(zip(ids, average)),
        "spearman_ab": _spearman(a, b),
        "spearman_a_observed": _spearman(a, actual),
        "spearman_b_observed": _spearman(b, actual),
        "spearman_mean_observed": _spearman(average, actual),
        "topk_a": rankings["a"],
        "topk_b": rankings["b"],
        "topk_mean": rankings["mean"],
        "topk_observed": rankings["observed"],
        "topk_overlap_fraction": len(a_set & b_set) / effective_k if effective_k else None,
        "topk_overlap_random_baseline": effective_k / n if n else None,
        "topk_jaccard": len(a_set & b_set) / len(a_set | b_set) if a_set | b_set else None,
        "observed_mean_all": all_mean,
        "observed_mean_topk_mean": selected_mean,
        "observed_topk_lift": selected_mean - all_mean if n else None,
        "oracle_capture_ratio": oracle_capture,
        "vector_stats": {
            "risk_a": _vector_stats(a),
            "risk_b": _vector_stats(b),
            "risk_mean": _vector_stats(average),
            "observed": _vector_stats(actual),
        },
    }
