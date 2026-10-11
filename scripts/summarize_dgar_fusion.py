#!/usr/bin/env python3
"""CPU analysis of paired fusion diagnostics; never report new-method accuracy.

Query routing selects among whole-model alpha passes. It measures whether prefix
features predict replay utility, NOT the performance of entity-wise fusion.
Fit stages select thresholds, directions, and the shared alpha; later stages
remain held out. All controls use checkpoint-alpha reference feature values.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


FEATURES = {
    "role_evolution": "prefix_subject_role_JS_excess",
    "neighbor_churn": "prefix_subject_neighbor_churn_excess",
    "frequency": "prefix_subject_frequency",
    "recency": "subject_recency",
    "norm_ratio": "subject_HCP_current_norm_ratio",
}


def numeric(value):
    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite diagnostic value")
    return value


def feature(query, name):
    reference = query["reference"]
    if name in ("role_evolution", "neighbor_churn"):
        if reference["prefix_subject_evolution_reliable"] != "True":
            return None
    return numeric(reference[FEATURES[name]])


def macro(queries, choose):
    by_stage = defaultdict(list)
    for query in queries:
        by_stage[query["stage"]].append(query["RR"][choose(query)])
    return statistics.mean(statistics.mean(values) for values in by_stage.values())


def rule_choice(query, rule, fallback):
    value = feature(query, rule["feature"])
    if value is None:
        return fallback
    high = value > rule["threshold"]
    return "0.75" if high == rule["high_prefers_current"] else "0.25"


def fit_rule(queries, name, fallback):
    values = [value for query in queries if (value := feature(query, name)) is not None]
    if not values:
        return None
    threshold = statistics.median(values)
    variants = [{"feature": name, "threshold": threshold, "high_prefers_current": direction}
                for direction in (True, False)]
    return max(variants, key=lambda rule: macro(
        queries, lambda query: rule_choice(query, rule, fallback)))


def bootstrap(queries, rule, fallback, count, seed, blocks):
    groups = defaultdict(lambda: defaultdict(list))
    for query in queries:
        delta = query["RR"][rule_choice(query, rule, fallback)] - query["RR"][fallback]
        groups[query["stage"]][query["cluster"]].append(delta)
    block_ids = sorted(blocks)
    rng, estimates = random.Random(seed), []
    for _ in range(count):
        block_means = []
        for block in rng.choices(block_ids, k=len(block_ids)):
            stage_means = []
            for stage in blocks[block]:
                clusters = list(groups[stage].values())
                drawn = rng.choices(clusters, k=len(clusters))
                stage_means.append(statistics.mean(value for cluster in drawn for value in cluster))
            block_means.append(statistics.mean(stage_means))
        estimates.append(statistics.mean(block_means))
    estimates.sort()
    return {"delta_ci025": estimates[int(.025 * (count - 1))],
            "delta_ci975": estimates[int(.975 * (count - 1))],
            "resampling": "temporal window block then original-fact clusters within each day",
            "warning": "Few held-out windows; descriptive uncertainty, not confirmatory evidence"}


def analyse(input_dir, fit_stages, eval_stages, samples, seed, eval_blocks=None):
    status = json.loads((input_dir / "status.json").read_text())
    if status.get("complete") is not True:
        raise ValueError("Fusion diagnostic did not complete")
    path = input_dir / "query_metrics.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    matched = defaultdict(dict)
    directions = defaultdict(set)
    for row in rows:
        key = (int(row["snapshot"]), row["fact_cluster_id"], row["direction"])
        directions[key[:2]].add(row["direction"])
        if row["alpha_label"] in matched[key]:
            raise ValueError("Duplicate query/alpha")
        matched[key][row["alpha_label"]] = row
    if any(value != {"forward", "inverse"} for value in directions.values()):
        raise ValueError("Every fact cluster must include forward and inverse queries")
    queries = []
    expected_labels = None
    for (stage, cluster, direction), conditions in matched.items():
        labels = set(conditions)
        expected_labels = labels if expected_labels is None else expected_labels
        if labels != expected_labels or not {"checkpoint", "0.25", "0.75"} <= labels:
            raise ValueError("Missing alpha condition or incomplete pairing")
        reference = conditions["checkpoint"]
        for row in conditions.values():
            if any(row[key] != reference[key] for key in ("subject", "relation", "object")):
                raise ValueError("Query identity differs across alpha conditions")
        queries.append({"stage": stage, "cluster": cluster, "direction": direction,
                        "reference": reference,
                        "RR": {label: 1 / float(row["rank"]) for label, row in conditions.items()}})
    actual = {query["stage"] for query in queries}
    if not queries or set(fit_stages) & set(eval_stages):
        raise ValueError("Need queries and disjoint fit/evaluation stages")
    if not set(fit_stages + eval_stages) <= actual or max(fit_stages) >= min(eval_stages):
        raise ValueError("Requested stages must exist and fit must precede evaluation")
    if eval_blocks is None:
        eval_blocks = {stage: [stage] for stage in eval_stages}
    block_days = [day for days in eval_blocks.values() for day in days]
    if len(block_days) != len(set(block_days)) or set(block_days) != set(eval_stages):
        raise ValueError("Evaluation blocks must cover each held-out day exactly once")
    fit = [query for query in queries if query["stage"] in fit_stages]
    held = [query for query in queries if query["stage"] in eval_stages]
    # Tie order preserves checkpoint-alpha before counterfactual labels.
    labels = ["checkpoint"] + sorted(expected_labels - {"checkpoint"}, key=float)
    global_alpha = max(labels, key=lambda label: macro(fit, lambda query: label))
    baseline = macro(held, lambda query: global_alpha)
    rules = {}
    for name in FEATURES:
        rule = fit_rule(fit, name, global_alpha)
        if rule is not None:
            value = macro(held, lambda query: rule_choice(query, rule, global_alpha))
            rules[name] = {**rule,
                           "fit_macro_rr": macro(fit, lambda query: rule_choice(query, rule, global_alpha)),
                           "heldout_macro_rr": value, "heldout_delta_vs_frozen_global": value - baseline,
                           **bootstrap(held, rule, global_alpha, samples, seed, eval_blocks)}
    stage_rows, strata_rows = [], []
    primary = rules.get("role_evolution")
    controls = {name: statistics.median(values) for name in ("frequency", "recency", "norm_ratio")
                if (values := [value for query in fit if (value := feature(query, name)) is not None])}
    for block, days in [*eval_blocks.items(), ("all", eval_stages)]:
        subset = [query for query in held if query["stage"] in days]
        baseline_stage = macro(subset, lambda query: global_alpha)
        rule_stage = macro(subset, lambda query: rule_choice(query, primary, global_alpha)) if primary else None
        stage_rows.append({"evaluation_block": block, "first_snapshot": min(days),
                           "last_snapshot": max(days), "query_count": len(subset),
                           "fact_clusters": len({query["cluster"] for query in subset}),
                           "frozen_global_alpha": global_alpha, "frozen_global_MRR": baseline_stage,
                           "role_rule_MRR_diagnostic_only": rule_stage,
                           "role_rule_delta": rule_stage - baseline_stage if primary else None})
        if primary:
            for name, threshold in [("all", None), *controls.items()]:
                bands = ("all",) if name == "all" else ("low", "high", "missing")
                for band in bands:
                    for evolution_band in ("low", "high"):
                        selected = []
                        for query in subset:
                            value = feature(query, "role_evolution")
                            if value is None or (value > primary["threshold"]) != (evolution_band == "high"):
                                continue
                            control = feature(query, name) if name != "all" else None
                            if name != "all":
                                query_band = "missing" if control is None else "high" if control > threshold else "low"
                                if query_band != band:
                                    continue
                            selected.append(query)
                        if selected:
                            clusters = len({query["cluster"] for query in selected})
                            strata_rows.append({"evaluation_block": block, "control": name, "control_band": band,
                                "evolution_band": evolution_band, "query_count": len(selected),
                                "fact_clusters": clusters, "enough_facts_30": clusters >= 30,
                                "preference_weighting": "query_pooled_descriptive",
                                "mean_current_preference": statistics.mean(
                                    query["RR"]["0.75"] - query["RR"]["0.25"] for query in selected)})
    summary = {"protocol": "retrospective_functional_predictability_screen_v1",
               "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "fit_snapshots": fit_stages, "evaluation_snapshots": eval_stages,
               "evaluation_window_blocks": eval_blocks,
               "frozen_global_alpha": global_alpha, "heldout_frozen_global_macro_MRR": baseline,
               "rules": rules, "control_thresholds_fit_only": controls,
               "rules_are_diagnostic_not_entity_gate_performance": True,
               "caveats": [
                   "Whole-model alpha passes selected per query are not entity-wise ECRI inference.",
                   "Final checkpoint may have learned future training facts; no online forecast claim.",
                   "Reference checkpoint-alpha norm controls only; no post-treatment norm controls.",
                   "Separate frequency/recency/norm stratifications are exploratory, not joint confounding control.",
                   "Stratum current-preference means are query-pooled descriptions; main RR results are equal-day macro.",
                   "A reverse high-evolution preference fails the prespecified current-preference hypothesis.",
                   "Raw/permutation evolution estimates are exploratory; no multiple-testing correction.",
                   "Do not select whichever rule wins held-out stages; role_evolution is preselected primary.",
                   "Stage signs and >=30 fact support must be checked; this script does not auto approve a route."]}
    return summary, stage_rows, strata_rows


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-snapshots", default="60,90,120")
    parser.add_argument("--eval-snapshots", default="180,240,300,360")
    parser.add_argument("--snapshot-radius", type=int, default=0,
                        help="Expand fit/evaluation center indices into fixed surrounding windows")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.bootstrap < 40:
        parser.error("Use at least 40 bootstrap samples")
    if args.snapshot_radius < 0:
        parser.error("snapshot-radius must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory")
    fit_centers = sorted(set(map(int, args.fit_snapshots.split(","))))
    eval_centers = sorted(set(map(int, args.eval_snapshots.split(","))))
    available = json.loads((args.input_dir / "status.json").read_text())["snapshots"]
    def expand(center):
        return list(range(max(2, center - args.snapshot_radius),
                          min(max(available), center + args.snapshot_radius) + 1))
    fit = sorted({day for center in fit_centers for day in expand(center)})
    blocks = {center: expand(center) for center in eval_centers}
    held = sorted({day for days in blocks.values() for day in days})
    summary, stages, strata = analyse(args.input_dir, fit, held, args.bootstrap, args.seed, blocks)
    summary["fit_snapshot_centers"] = fit_centers
    summary["evaluation_snapshot_centers"] = eval_centers
    summary["snapshot_radius"] = args.snapshot_radius
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output_dir / "window_comparison.csv", stages)
    write_csv(args.output_dir / "controlled_strata.csv", strata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
