"""Temporary mechanism measurements that leave the training trajectory intact.

Candidate CE is measured under the current stage's history, without diffusion.
The actual DGAR target concerns post-training weights under this same scoring
context; it is not a full-pipeline historical MRR or a causal replay benefit.
"""

from __future__ import annotations

import random
import time

from diffusion.fahc_selector import positive_loss_increase
from src.fahc import (_candidate_losses, _capture_runtime_state,
                      _restore_runtime_state, probe_hcp_risks)
from src.fahc_mechanism_metrics import summarize_mechanism
from src.fahc_runtime import isolated_rng


def _summarize(ids, a, b, observed, k, seed):
    return summarize_mechanism(ids, {index: a[index] for index in ids},
                               {index: b[index] for index in ids},
                               {index: observed[index] for index in ids}, k, seed)


def _configured_prediction(metric, ids, risk, observed, k, seed):
    # The duplicated vector is only used to reuse ranking calculations;
    # no stability coefficient from this single-probe comparison is exported.
    configured = _summarize(ids, risk, risk, observed, k, seed)
    metric.update(
        configured_probe_spearman_observed=configured["spearman_mean_observed"],
        configured_probe_observed_topk_lift=configured["observed_topk_lift"])


def begin_mechanism_diagnostic(args, snapshot, model, old_model, candidates,
                               current, history, static_graph, use_cuda,
                               training_optimizer):
    """Compare disjoint mini-batches from the same pre-stage state."""
    started = time.perf_counter()
    current_count = len(current)
    batch_facts = min(args.fahc_probe_facts, current_count // 2)
    if batch_facts < 2:
        raise ValueError("mechanism diagnostic needs at least four current training facts")
    seed = args.seed + 13000019 * snapshot
    positions = random.Random(seed).sample(range(current_count), 2 * batch_facts)
    position_a = sorted(positions[:batch_facts])
    position_b = sorted(positions[batch_facts:])
    device = next(model.parameters()).device

    def measure(batch, steps):
        detail = {}
        # Match dropout RNG between A/B to isolate the sampled data change.
        with isolated_rng(seed, device):
            risks = probe_hcp_risks(
                model, old_model, candidates, batch, history, static_graph,
                use_cuda, args.task_weight,
                args.fahc_probe_lr if args.fahc_probe_lr is not None else args.lr,
                args.grad_norm, steps, args.fahc_score_batch_size,
                training_optimizer=training_optimizer, diagnostics=detail)
        return risks, detail

    risk_a, detail_a = measure(current[position_a], args.fahc_probe_steps)
    risk_b, detail_b = measure(current[position_b], args.fahc_probe_steps)
    configured_positions = sorted(random.Random(seed + 1).sample(
        range(current_count), min(args.fahc_probe_facts, current_count)))
    with isolated_rng(seed, device):
        configured_risk = probe_hcp_risks(
            model, old_model, candidates, current[configured_positions], history,
            static_graph, use_cuda, args.task_weight,
            args.fahc_probe_lr if args.fahc_probe_lr is not None else args.lr,
            args.grad_norm, args.fahc_probe_steps, args.fahc_score_batch_size,
            training_optimizer=training_optimizer)
    # Reference update uses the trainer's LR, not an optional probe LR.
    reference_detail = {}
    with isolated_rng(seed, device):
        observed = probe_hcp_risks(
            model, old_model, candidates, current, history, static_graph,
            use_cuda, args.task_weight, args.lr, args.grad_norm,
            args.fahc_mechanism_reference_steps, args.fahc_score_batch_size,
            training_optimizer=training_optimizer, diagnostics=reference_detail)
    nonempty = [candidate.time_index for candidate in candidates if candidate.triples]
    metrics = {
        "current_only": _summarize(
            nonempty, risk_a, risk_b, observed, args.reply_batch, seed),
        "fixed_bn_current_only": _summarize(
            nonempty, detail_a["fixed_bn_risks"], detail_b["fixed_bn_risks"],
            reference_detail["fixed_bn_risks"], args.reply_batch, seed),
    }
    _configured_prediction(metrics["current_only"], nonempty, configured_risk,
                           observed, args.reply_batch, seed)
    facts_a = {tuple(row[:3]) for row in current[position_a].detach().cpu().tolist()}
    facts_b = {tuple(row[:3]) for row in current[position_b].detach().cpu().tolist()}
    record = {
        "event": "mechanism", "snapshot": snapshot,
        "protocol": "disjoint_current_batches_then_current_only_and_actual_dgar",
        "score_context": "history_only_current_stage_entity_ce",
        "batch_facts": batch_facts, "requested_batch_facts": args.fahc_probe_facts,
        "current_fact_count": current_count, "batch_a_positions": position_a,
        "batch_b_positions": position_b, "matched_dropout_seed": seed,
        "batch_sampling": "disjoint_rows_without_replacement",
        "batch_fact_overlap_count": len(facts_a & facts_b),
        "configured_probe_facts": len(configured_positions),
        "configured_probe_positions": configured_positions,
        "probe_steps": args.fahc_probe_steps,
        "reference_steps": args.fahc_mechanism_reference_steps,
        "candidate_count": len(candidates), "nonempty_candidate_count": len(nonempty),
        "candidate_indices": [candidate.time_index for candidate in candidates],
        "candidate_fact_counts": {candidate.time_index: len(candidate.triples)
                                  for candidate in candidates},
        "risk_a": risk_a, "risk_b": risk_b,
        "risk_configured_probe": configured_risk,
        "risk_a_fixed_bn": detail_a["fixed_bn_risks"],
        "risk_b_fixed_bn": detail_b["fixed_bn_risks"],
        "observed_current_only": observed,
        "observed_current_only_signed": reference_detail["signed_increase"],
        "metrics": metrics,
        "temporary_diagnostic_seconds": time.perf_counter() - started,
    }
    return {"record": record, "candidates": candidates,
            "old_losses": detail_a["old_losses"], "nonempty": nonempty,
            "seed": seed}


def finish_mechanism_diagnostic(args, model, pending, history, static_graph,
                                use_cuda):
    """Measure the stage's actual best DGAR weights, preserving model/RNG."""
    runtime = _capture_runtime_state(model)
    device = next(model.parameters()).device
    try:
        with isolated_rng(pending["seed"], device):
            after = _candidate_losses(model, pending["candidates"], history,
                                      static_graph, use_cuda,
                                      args.fahc_score_batch_size)
    finally:
        _restore_runtime_state(runtime)
    old = pending["old_losses"]
    observed = {index: positive_loss_increase(values, after[index])
                for index, values in old.items()}
    record = pending["record"]
    record["observed_dgar_history_only"] = observed
    record["observed_dgar_history_only_signed"] = {
        index: (sum(new - previous for previous, new in zip(values, after[index]))
                / len(values) if values else 0.0)
        for index, values in old.items()}
    record["metrics"]["dgar_history_only"] = _summarize(
        pending["nonempty"], record["risk_a"], record["risk_b"], observed,
        args.reply_batch, pending["seed"])
    _configured_prediction(record["metrics"]["dgar_history_only"],
                           pending["nonempty"], record["risk_configured_probe"],
                           observed, args.reply_batch, pending["seed"])
    return record
