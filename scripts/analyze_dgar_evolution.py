#!/usr/bin/env python3
"""Training-data evolution diagnostics, using only the Python standard library.

This tool does not load a model or start training. ``current-dgar`` reconstructs
main.py's actual split: concatenate the time snapshots from the three source
files, then retain their first 80% as training facts. The held-out suffixes never
contribute to the statistics. ``canonical-train`` uses only raw train.txt.
Every transition estimate uses its completed training prefix; an estimate for
window j is available for decisions after window j, never before it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_snapshots(path):
    snapshots, current, previous = [], [], None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{path}:{number}: expected s r o timestamp")
            subject, relation, object_, timestamp = map(int, parts[:4])
            if min(subject, relation, object_) < 0:
                raise ValueError(f"{path}:{number}: negative identifier")
            if previous is not None and timestamp < previous:
                raise ValueError(f"{path}:{number}: timestamps are not sorted")
            if previous is not None and timestamp != previous:
                snapshots.append((previous, current))
                current = []
            current.append((subject, relation, object_))
            previous = timestamp
    if current:
        snapshots.append((previous, current))
    return snapshots


def js_divergence(left, right):
    """Jensen-Shannon divergence in bits, in [0, 1]."""
    total_left, total_right = sum(left.values()), sum(right.values())
    if not total_left or not total_right:
        return None
    result = 0.0
    for key in left.keys() | right.keys():
        a, b = left[key] / total_left, right[key] / total_right
        midpoint = (a + b) / 2
        if a:
            result += 0.5 * a * math.log2(a / midpoint)
        if b:
            result += 0.5 * b * math.log2(b / midpoint)
    return max(0.0, min(1.0, result))


def permutation_summary(observed, null):
    return {
        "js": observed,
        "null_mean": statistics.mean(null),
        "js_excess": observed - statistics.mean(null),
        "permutation_p": (1 + sum(value >= observed for value in null)) / (len(null) + 1),
    }


def tail_test(old_tails, new_tails, sample_cap, permutations, rng):
    count = min(sample_cap, len(old_tails), len(new_tails))
    left, right = rng.sample(old_tails, count), rng.sample(new_tails, count)
    observed = js_divergence(Counter(left), Counter(right))
    pooled, null = left + right, []
    for _ in range(permutations):
        rng.shuffle(pooled)
        null.append(js_divergence(Counter(pooled[:count]), Counter(pooled[count:])))
    return count, permutation_summary(observed, null)


def conditional_test(old_queries, new_queries, query_cap, query_min, permutations, rng):
    """Control subject composition by balancing each repeated (subject, r).

    Randomization shuffles window labels within a subject, so the null retains
    each query's object distribution and matched support. Low support remains
    flagged; a large uncorrected JS alone is not evolution evidence.
    """
    pairs = []
    for subject in sorted(old_queries.keys() & new_queries.keys()):
        count = min(query_cap, len(old_queries[subject]), len(new_queries[subject]))
        if count >= query_min:
            pairs.append((subject, rng.sample(old_queries[subject], count),
                          rng.sample(new_queries[subject], count)))
    if not pairs:
        return 0, 0, None
    left = Counter((s, tail) for s, a, _ in pairs for tail in a)
    right = Counter((s, tail) for s, _, b in pairs for tail in b)
    observed, null = js_divergence(left, right), []
    for _ in range(permutations):
        random_left, random_right = Counter(), Counter()
        for subject, a, b in pairs:
            pooled = a + b
            rng.shuffle(pooled)
            size = len(a)
            random_left.update((subject, tail) for tail in pooled[:size])
            random_right.update((subject, tail) for tail in pooled[size:])
        null.append(js_divergence(random_left, random_right))
    return len(pairs), sum(len(a) for _, a, _ in pairs), permutation_summary(observed, null)


def distribution_summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]}


def analyze(args):
    started = time.perf_counter()
    source_names = ("train", "valid", "test") if args.split == "current-dgar" else ("train",)
    inputs, snapshots = [], []
    for split in source_names:
        path = args.data_dir / f"{split}.txt"
        loaded = read_snapshots(path)
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path),
                       "source_role": split, "source_snapshots": len(loaded),
                       "source_facts": sum(len(facts) for _, facts in loaded)})
        snapshots.extend(loaded)
    timestamps = [timestamp for timestamp, _ in snapshots]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("source snapshot concatenation is not strictly chronological")
    source_snapshot_count = len(snapshots)
    if args.prefix_snapshots:
        snapshots = snapshots[:args.prefix_snapshots]
    if len(snapshots) < args.windows:
        raise ValueError("training prefix must contain at least one snapshot per window")
    stat_path = args.data_dir / "stat.txt"
    relation_count = int(stat_path.read_text().split()[1])
    inputs.append({"path": str(stat_path.resolve()), "sha256": sha256(stat_path),
                   "source_role": "identifier_counts"})
    retained = []
    for timestamp, facts in snapshots:
        count = int(0.8 * len(facts)) if args.split == "current-dgar" else len(facts)
        train_facts = facts[:count]
        if any(relation >= relation_count for _, relation, _ in train_facts):
            raise ValueError("source relation id exceeds stat.txt relation count")
        retained.append((timestamp, train_facts))
    window_snapshots = [[] for _ in range(args.windows)]
    for index, snap in enumerate(retained):
        window_snapshots[index * args.windows // len(retained)].append(snap)
    records, windows, relation_histograms, entity_histograms = [], [], [], []
    directed_queries = []
    for index, window in enumerate(window_snapshots):
        facts = [fact for _, snapshot in window for fact in snapshot]
        rel_counts = Counter(relation for _, relation, _ in facts)
        ent_counts = Counter(entity for subject, _, object_ in facts for entity in (subject, object_))
        queries = defaultdict(lambda: defaultdict(list))
        for subject, relation, object_ in facts:
            queries[relation][subject].append(object_)
            if args.inverse:
                queries[relation + relation_count][object_].append(subject)
        directed_queries.append(queries)
        relation_histograms.append(rel_counts)
        entity_histograms.append(ent_counts)
        windows.append({"window": index + 1, "snapshot_count": len(window),
                        "first_timestamp": window[0][0], "last_timestamp": window[-1][0],
                        "training_fact_count": len(facts), "unique_relations": len(rel_counts),
                        "unique_entities": len(ent_counts),
                        "estimate_available_after_timestamp": window[-1][0]})
    transitions = []
    directions = 2 if args.inverse else 1
    for transition in range(args.windows - 1):
        old, new = directed_queries[transition], directed_queries[transition + 1]
        transition_rows = []
        for relation in range(relation_count * directions):
            old_queries, new_queries = old.get(relation, {}), new.get(relation, {})
            old_tails = [tail for values in old_queries.values() for tail in values]
            new_tails = [tail for values in new_queries.values() for tail in values]
            common = old_queries.keys() & new_queries.keys()
            old_pairs = {(subject, tail) for subject, tails in old_queries.items() for tail in tails}
            new_pairs = {(subject, tail) for subject, tails in new_queries.items() for tail in tails}
            union = old_pairs | new_pairs
            row = {
                "old_window": transition + 1, "new_window": transition + 2,
                "estimate_available_after_timestamp": windows[transition + 1]["last_timestamp"],
                "relation_id": relation, "base_relation_id": relation % relation_count,
                "direction": "inverse" if relation >= relation_count else "forward",
                "old_count": len(old_tails), "new_count": len(new_tails),
                "old_subject_count": len(old_queries), "new_subject_count": len(new_queries),
                "repeated_query_count": len(common),
                "old_repeated_query_fact_fraction": (sum(len(old_queries[s]) for s in common) / len(old_tails)
                                                     if old_tails else None),
                "new_repeated_query_fact_fraction": (sum(len(new_queries[s]) for s in common) / len(new_tails)
                                                     if new_tails else None),
                "subject_overlap_jaccard": (len(common) / len(old_queries.keys() | new_queries.keys())
                                            if old_queries or new_queries else None),
                "event_overlap_jaccard": len(old_pairs & new_pairs) / len(union) if union else None,
                "tail_unreliable": min(len(old_tails), len(new_tails)) < args.min_count,
                "tail_sample_count_each": 0, "tail_js": None, "tail_null_mean": None,
                "tail_js_excess": None, "tail_permutation_p": None,
                "conditional_query_count": 0, "conditional_sample_count_each": 0,
                "conditional_unreliable": True, "conditional_js": None,
                "conditional_null_mean": None, "conditional_js_excess": None,
                "conditional_permutation_p": None,
            }
            rng = random.Random(args.seed + 1000003 * transition + 1009 * relation)
            if not row["tail_unreliable"]:
                count, result = tail_test(old_tails, new_tails, args.max_sample, args.permutations, rng)
                row["tail_sample_count_each"] = count
                row.update({f"tail_{key}": value for key, value in result.items()})
            conditional_rng = random.Random(args.seed + 1000003 * transition
                                            + 1009 * relation + 50000947)
            query_count, support, conditional = conditional_test(
                old_queries, new_queries, args.query_cap, args.query_min,
                args.permutations, conditional_rng)
            row["conditional_query_count"] = query_count
            row["conditional_sample_count_each"] = support
            row["conditional_unreliable"] = (query_count < args.min_common_queries
                                             or support < args.min_count)
            if conditional is not None:
                row.update({f"conditional_{key}": value for key, value in conditional.items()})
            transition_rows.append(row)
        records.extend(transition_rows)
        reliable = [row for row in transition_rows if not row["tail_unreliable"]]
        conditional_reliable = [row for row in transition_rows if not row["conditional_unreliable"]]
        forward = [row for row in reliable if row["direction"] == "forward"]
        conditional_forward = [row for row in conditional_reliable if row["direction"] == "forward"]
        transitions.append({
            "old_window": transition + 1, "new_window": transition + 2,
            "relation_frequency_js_bits": js_divergence(relation_histograms[transition], relation_histograms[transition + 1]),
            "entity_frequency_js_bits": js_divergence(entity_histograms[transition], entity_histograms[transition + 1]),
            "reliable_forward_tail_relations": len(forward),
            "forward_tail_js_excess": distribution_summary(row["tail_js_excess"] for row in forward),
            "forward_tail_p_at_most_005_count": sum(row["tail_permutation_p"] <= 0.05 for row in forward),
            "reliable_forward_conditional_relations": len(conditional_forward),
            "forward_conditional_js_excess": distribution_summary(row["conditional_js_excess"] for row in conditional_forward),
            "forward_conditional_p_at_most_005_count": sum(row["conditional_permutation_p"] <= 0.05
                                                           for row in conditional_forward),
            "reliable_directed_tail_relations": len(reliable),
            "reliable_directed_conditional_relations": len(conditional_reliable),
        })
    cache_paths = sorted((args.data_dir / "history_snap_v2").glob("snap_*.pkl"),
                         key=lambda path: int(path.stem.split("_")[-1]))
    cache_digest = hashlib.sha256()
    for path in cache_paths:
        cache_digest.update(f"{path.name}:{sha256(path)}\n".encode())
    summary = {
        "tool": "analyze_dgar_evolution.py", "model_or_gpu_executed": False,
        "units": "Jensen-Shannon divergence in bits; proportions in 0..1",
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "split_definition": ("concatenate raw train/valid/test time snapshots in that order; each snapshot's "
                             "first floor(0.8*N) facts are training, remaining suffix excluded"
                             if args.split == "current-dgar" else "raw train.txt facts only"),
        "source_inputs": inputs, "source_snapshot_count": source_snapshot_count,
        "used_training_snapshot_count": len(retained),
        "training_fact_count": sum(len(facts) for _, facts in retained),
        "prefix_causality": "each estimate uses only training windows up to its new_window endpoint; "
                            "use for later updates only, not retrospectively within that window",
        "window_definition": "chronological index * windows // retained_snapshot_count",
        "inverse_definition": "source forward r and constructed inverse r+num_relations analyzed separately; "
                              "entity/relation global frequencies use source facts once",
        "null_definition": "tail: pooled equal-count sample labels shuffled; conditional: labels shuffled "
                           "within each repeated subject query with equal samples per side",
        "interpretation_limits": [
            "Exploratory data statistics, not neural representation drift or causal forgetting evidence.",
            "Tail changes include subject composition changes; inspect conditional support and null comparison.",
            "No multiple-comparison correction; permutation p values are descriptive screening signals.",
            "Current-DGAR raw valid/test filenames contain facts reassigned to training by existing code; "
            "only reassigned training prefixes are used, never the actual held-out suffixes.",
            "Frequency JS over 230 relations and 7128 entities is not a fair dimension-controlled comparison.",
            "Cache hash records bytes only; this standard-library tool does not unpickle or validate cache semantics.",
        ],
        "cache_manifest": {"files": len(cache_paths), "sha256_name_and_file_hash_manifest": cache_digest.hexdigest()},
        "windows": windows, "transitions": transitions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "relation_windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "training_snapshots": len(retained),
                      "training_facts": summary["training_fact_count"], "relation_rows": len(records),
                      "permutations": args.permutations, "elapsed_seconds": round(summary["elapsed_seconds"], 2)}))
    return summary


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=repo / "data" / "ICEWS14s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("current-dgar", "canonical-train"), default="current-dgar")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--prefix-snapshots", type=int, default=0, help="0 uses all training snapshots")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--min-count", type=int, default=80)
    parser.add_argument("--max-sample", type=int, default=200)
    parser.add_argument("--query-min", type=int, default=3)
    parser.add_argument("--query-cap", type=int, default=20)
    parser.add_argument("--min-common-queries", type=int, default=5)
    parser.add_argument("--no-inverse", dest="inverse", action="store_false")
    parser.set_defaults(inverse=True)
    args = parser.parse_args()
    for name in ("windows", "permutations", "min_count", "max_sample", "query_min", "query_cap", "min_common_queries"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.windows < 2 or args.prefix_snapshots < 0 or args.query_min > args.query_cap:
        parser.error("need windows >=2, prefix-snapshots >=0, and query-min <= query-cap")
    analyze(args)


if __name__ == "__main__":
    main()
