"""Configuration checks and reproducible experiment IDs for FAHC-DGAR.

This module intentionally has no model or data dependencies so invalid runs can
be rejected before loading a dataset or constructing a CUDA model.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path


ALGORITHM_VERSION = "full-hcp-v3"
HCP_MODES = frozenset({"original", "random", "recent", "risk", "coverage",
                       "risk+coverage"})
DIFFUSION_MODES = frozenset({"add", "Linear"})


def parse_bool(value: object) -> bool:
    """Parse a CLI boolean without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _real(args, name: str, *, minimum: float | None = None,
          maximum: float | None = None, strict_minimum: bool = False) -> float:
    value = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"--{name.replace('_', '-')} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"--{name.replace('_', '-')} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = ">" if strict_minimum else ">="
        raise ValueError(f"--{name.replace('_', '-')} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"--{name.replace('_', '-')} must be <= {maximum}")
    return result


def _integer(args, name: str, *, minimum: int = 0,
             maximum: int | None = None) -> int:
    value = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"--{name.replace('_', '-')} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"--{name.replace('_', '-')} must be <= {maximum}")
    return int(value)


def validate_args(args: object) -> None:
    """Fail fast on configurations that would crash or mislabel a FAHC run.

    Replay switches and diffusion mode are normalized in place, so callers
    that still receive legacy string booleans do not accidentally enable them
    through Python string truthiness.
    """
    selection = getattr(args, "hcp_selection")
    if selection not in HCP_MODES:
        raise ValueError(f"unknown HCP selection mode: {selection!r}")

    stages = getattr(args, "fahc_mechanism_stages", "")
    if isinstance(stages, str):
        try:
            stages = tuple(int(value.strip()) for value in stages.split(",")) if stages else ()
        except ValueError as exc:
            raise ValueError("--fahc-mechanism-stages expects comma-separated snapshot indices") from exc
    if (not isinstance(stages, (list, tuple)) or
            any(isinstance(value, bool) or not isinstance(value, Integral)
                or value < 2 for value in stages) or len(set(stages)) != len(stages)):
        raise ValueError("--fahc-mechanism-stages requires distinct integer indices >= 2")
    args.fahc_mechanism_stages = tuple(sorted(stages))
    args.fahc_mechanism_reference_steps = getattr(args, "fahc_mechanism_reference_steps", 10)
    _integer(args, "fahc_mechanism_reference_steps", minimum=1)
    if stages:
        if (selection != "random" or args.fahc_kd_weight != 0 or args.n_epochs != 1
                or getattr(args, "test", False) or getattr(args, "grid_search", False)):
            raise ValueError("mechanism diagnostic requires chronological random replay, KD0, and no --test/grid-search")
        if min(stages) <= args.reply_batch + 1:
            raise ValueError("mechanism stages must exceed k+1 to allow historical selection")

    for name in ("delete_feature_reply", "delete_his_prompt", "add_history"):
        if hasattr(args, name):
            setattr(args, name, parse_bool(getattr(args, name)))

    diffuc = getattr(args, "diffuc")
    if diffuc not in DIFFUSION_MODES:
        diffuc = "add" if parse_bool(diffuc) else False
    args.diffuc = diffuc
    diffusion_active = diffuc in DIFFUSION_MODES

    for name in ("n_epochs", "regcn_epochs", "diffu_epochs", "n_layers",
                 "train_history_len", "test_history_len", "dilate_len",
                 "n_hidden", "hidden_size", "attn_heads", "heads",
                 "num_blocks", "diffusion_steps", "max_len", "his_max_len",
                 "batch_size", "accumulation_steps", "fahc_time_bin_size",
                 "fahc_probe_steps", "fahc_probe_facts",
                 "fahc_score_batch_size"):
        _integer(args, name, minimum=1)
    for name in ("fahc_fact_budget", "fahc_eval_every",
                 "fahc_past_eval_samples", "start_reply", "num_blocks_cross"):
        _integer(args, name, minimum=0)

    _integer(args, "reply_batch", minimum=0)
    _integer(args, "seed", minimum=0, maximum=2**32 - 1)
    if hasattr(args, "reply_batch_num"):
        _integer(args, "reply_batch_num", minimum=1)

    _real(args, "lr", minimum=0, strict_minimum=True)
    _real(args, "grad_norm", minimum=0, strict_minimum=True)
    for name in ("mu", "mu_r", "weight", "lambda_uncertainty",
                 "pattern_noise_radio", "fahc_eta", "fahc_entity_weight",
                 "fahc_kd_weight"):
        _real(args, name, minimum=0)
    for name in ("task_weight", "fahc_kd_confidence"):
        _real(args, name, minimum=0, maximum=1)
    for name in ("dropout", "input_dropout", "hidden_dropout",
                 "feat_dropout", "emb_dropout"):
        _real(args, name, minimum=0, maximum=1)
    for name in ("angle", "classifier_scale"):
        _real(args, name)
    _real(args, "fahc_candidate_multiplier", minimum=1)
    _real(args, "fahc_kd_temperature", minimum=0, strict_minimum=True)
    _real(args, "temperature_object", minimum=0, strict_minimum=True)
    scale = _real(args, "scale", minimum=0, strict_minimum=True)
    beta_start = _real(args, "beta_start", minimum=0, maximum=1,
                       strict_minimum=True)
    beta_end = _real(args, "beta_end", minimum=0, maximum=1,
                     strict_minimum=True)
    if beta_start > beta_end:
        raise ValueError("--beta-start must not exceed --beta-end")
    if getattr(args, "fahc_probe_lr") is not None:
        _real(args, "fahc_probe_lr", minimum=0, strict_minimum=True)

    discount = _real(args, "discount")
    if discount not in (0, 1):
        raise ValueError("--discount must be 0 or 1; other values skip the static loss")
    if args.n_hidden % args.attn_heads or args.hidden_size % args.heads:
        raise ValueError("hidden sizes must be divisible by their attention head counts")

    if diffusion_active:
        if args.diffusion_steps < 2:
            raise ValueError("--diffusion-steps must be at least 2 for diffusion sampling")
        if getattr(args, "noise_schedule", None) == "linear" and scale * beta_end > 1:
            raise ValueError("linear diffusion requires --scale * --beta-end <= 1")
        if args.start_reply >= args.regcn_epochs:
            raise ValueError("--start-reply must be less than --regcn-epochs")
        if args.n_hidden != args.hidden_size:
            raise ValueError("--n-hidden must equal --hidden_size when diffusion is active")

    if args.fahc_kd_weight > 0 and selection == "original":
        raise ValueError("use --hcp-selection random for budget-matched HCP-KD")
    if (selection != "original" or getattr(args, "fahc_diagnostics", False)
            or getattr(args, "fahc_final_validation", False)) and args.n_epochs != 1:
        raise ValueError("FAHC diagnostics require --n-epochs 1 to preserve chronological stages")
    if selection != "original":
        if args.reply_batch < 1 or args.fahc_fact_budget < args.reply_batch:
            raise ValueError("controlled HCP selection needs fact budget >= k >= 1")
        if not diffusion_active or args.delete_feature_reply:
            raise ValueError("FAHC HCP selection requires diffusion feature replay")
        if getattr(args, "delete_his_prompt", False):
            raise ValueError("FAHC HCP selection requires historical HCP prompts")
        if hasattr(args, "add_history") and not args.add_history:
            raise ValueError("FAHC HCP selection requires historical context")


# These options affect reporting or where a run executes, not trained weights.
_SIGNATURE_EXCLUDE = frozenset({
    "gpu", "test", "grid_search", "tune", "run_analysis", "run_statistic",
    "multi_step", "topk", "num_k", "batch_size", "evaluate_every",
    "relation_evaluation", "test_history_len", "fahc_eval_every",
    "fahc_past_eval_samples", "fahc_diagnostics", "fahc_final_validation", "fahc_log", "vl",
    "log_dir", "save_dir",
    "output_dir", "model_dir", "checkpoint_dir", "model_state_file",
    "diffu_model_state_file", "num_nodes", "num_rels",
    "fahc_mechanism_stages", "fahc_mechanism_reference_steps",
})

_FLOAT_FIELDS = frozenset({
    "lr", "grad_norm", "mu", "mu_r", "weight", "task_weight", "discount",
    "angle", "dropout", "input_dropout", "hidden_dropout", "feat_dropout",
    "emb_dropout", "lambda_uncertainty", "pattern_noise_radio", "fahc_eta",
    "fahc_entity_weight", "fahc_candidate_multiplier", "fahc_probe_lr",
    "fahc_kd_weight", "fahc_kd_temperature", "fahc_kd_confidence",
    "temperature_object", "scale", "beta_start", "beta_end",
    "classifier_scale",
})


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment signature cannot contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported experiment setting type: {type(value).__name__}")


def experiment_settings(args: object) -> dict[str, object]:
    """Return the training/model settings used to identify a checkpoint."""
    settings: dict[str, object] = {"algorithm_version": ALGORITHM_VERSION}
    for name, value in vars(args).items():
        if name in _SIGNATURE_EXCLUDE or name.startswith("_"):
            continue
        if name == "fahc_probe_lr" and value is None:
            value = getattr(args, "lr")
        if name in _FLOAT_FIELDS and value is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be numeric for the experiment signature")
            value = float(value)
        settings[name] = _json_value(value)
    return settings


def experiment_signature(args: object) -> str:
    """Return a stable ten-hex-digit SHA-256 ID for a trained configuration."""
    payload = json.dumps(experiment_settings(args), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]
