"""Stateless routing of the direct entity-candidate gradient in replay CE."""
import math


def effective_candidate_alpha(mode="original", alpha=1.0, target="entity"):
    """Resolve CLI settings; relation-candidate routing is not implemented."""
    if mode not in ("original", "candidate-stopgrad", "candidate-scaled"):
        raise ValueError(f"Unknown QCDR mode: {mode}")
    if target != "entity":
        raise ValueError("QCDR currently supports only the entity candidate branch")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("qcdr-candidate-alpha must be finite and in [0, 1]")
    if mode == "original":
        return 1.0
    if mode == "candidate-stopgrad":
        return 0.0
    return float(alpha)


def validate_replay_entity_weight(weight):
    """Multiplier of BOTH historical entity CE terms, for the control ablation."""
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("qcdr-replay-entity-weight must be finite and in [0, 1]")
    return float(weight)


def decouple_candidate_gradient(candidate_repr, alpha):
    """Keep finite forward values, dtype/device and scale only this branch.

    Apply AFTER taking the subject/query view, immediately before candidate
    scoring. Shared parameters may still receive gradients through the query.
    The zero-valued residual avoids rounding the forward values by separately
    multiplying two copies of the candidate bank. Endpoints are exact bypasses.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("candidate gradient scale must be finite and in [0, 1]")
    if alpha == 1.0:
        return candidate_repr
    fixed = candidate_repr.detach()
    if alpha == 0.0:
        return fixed
    return fixed + alpha * (candidate_repr - fixed)
