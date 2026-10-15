"""Optional, passive PGR observations; never feed these labels back to training.

Evaluation follows Original DGAR's retrospective protocol. 'seen_before' uses
only training facts before the evaluated query's time, and is a reporting label,
not a restriction on the author's pre-existing diffusion guidance history.
"""
import csv
import json
import math
import statistics
import time

import torch


FIELDS = ("stage", "subject", "relation", "target", "inverse", "seen_before",
          "rank_raw", "rank_filtered")


@torch.no_grad()
def adapter_statistics(model, features):
    adapter = getattr(model, "pgr_adapter", None)
    result = {"mode": getattr(model.args, "pgr_mode", "original"),
              "parameters": 0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
              "facts": 0 if features is None else len(features)}
    scalars = {}
    if adapter is not None:
        scalars.update({name + "_norm": value.detach().float().norm()
                        for name, value in adapter.named_parameters()})
    if features is not None and len(features):
        head, tail = features[:, 0].detach(), features[:, 2].detach()
        adapted_h, adapted_t = (head, tail) if adapter is None else (adapter(head), adapter(tail))
        for name, value in (("input_head", head), ("input_tail", tail),
                            ("adapted_head", adapted_h), ("adapted_tail", adapted_t),
                            ("head_residual", adapted_h - head), ("tail_residual", adapted_t - tail)):
            scalars[name + "_mean_norm"] = value.float().norm(dim=-1).mean()
    # One optional transfer per observation, rather than per entity/forward.
    if scalars:
        result.update(zip(scalars, torch.stack(list(scalars.values())).cpu().tolist()))
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError("non-finite PGR diagnostic statistic")
    return result


def append_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def record_training_stage(args, stage, model, features):
    runtime = getattr(args, "_pgr_runtime", None)
    geometry = (runtime.catalog.get(stage, {}).get("stats", {}) if runtime else {})
    append_json(args.run_dir / "diagnostics" / "stages.jsonl",
                {"event": "best_stage", "stage": stage,
                 "protocol": args._provenance["protocol"], "geometry": geometry,
                 **adapter_statistics(model, features)})


def metrics(rows):
    if not rows:
        return {"queries": 0, "mrr": None, "hits1": None, "hits3": None, "hits10": None}
    ranks = [int(row["rank_filtered"]) for row in rows]
    return {"queries": len(rows), "mrr": statistics.fmean(1 / r for r in ranks),
            **{f"hits{k}": statistics.fmean(r <= k for r in ranks) for k in (1, 3, 10)}}


def summarize_rows(rows, final_stage):
    buckets = {}
    for row in rows:
        buckets.setdefault(int(row["stage"]), []).append(row)
    stages = sorted(buckets)
    by_stage = {s: metrics(buckets[s]) for s in stages}
    past = [by_stage[s] for s in stages if 1 <= s < final_stage]
    groups = {name: metrics([r for r in rows if int(r[key]) == value])
              for name, key, value in (("forward", "inverse", 0), ("inverse", "inverse", 1),
                                        ("repeated", "seen_before", 1), ("unseen", "seen_before", 0))}
    return {"macro_mrr": statistics.fmean(m["mrr"] for m in past) if past else None,
            "pooled": metrics(rows), "current": by_stage.get(final_stage),
            "groups": groups, "stages": by_stage}


class EvaluationRecorder:
    """Fresh output directory on each evaluation, with no checkpoint mutation."""

    def __init__(self, args, phase, history_list, num_relations):
        if phase not in ("test", "validation"):
            raise ValueError("unsupported diagnostic split")
        self.directory = args.run_dir / "diagnostics" / f"{phase}-{time.time_ns()}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.args, self.phase, self.history = args, phase, history_list
        self.num_relations, self.prefix = num_relations, 0
        self.seen, self.rows, self.visited = set(), [], []
        with (self.directory / "queries.csv").open("x", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def record(self, stage, triples, rank_raw, rank_filtered, model, features):
        if self.visited and stage <= self.visited[-1]:
            raise ValueError("diagnostic stages must increase")
        while self.prefix < stage:
            for s, r, o in self.history[self.prefix]:
                s, r, o = int(s), int(r), int(o)
                inverse_r = r + self.num_relations if r < self.num_relations else r - self.num_relations
                self.seen.update(((s, r, o), (o, inverse_r, s)))
            self.prefix += 1
        tags = triples.detach().cpu().tolist()
        raw, filtered = rank_raw.detach().cpu().tolist(), rank_filtered.detach().cpu().tolist()
        if len(tags) != len(raw) or len(tags) != len(filtered):
            raise ValueError("query/rank lengths differ")
        rows = []
        for (s, r, o), a, b in zip(tags, raw, filtered):
            if a < 1 or b < 1 or int(a) != a or int(b) != b:
                raise ValueError("ranks must be positive integers")
            rows.append(dict(zip(FIELDS, (stage, s, r, o, int(r >= self.num_relations),
                                         int((s, r, o) in self.seen), int(a), int(b)))))
        with (self.directory / "queries.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerows(rows)
        self.rows.extend(rows)
        self.visited.append(stage)
        append_json(self.directory / "replay.jsonl",
                    {"event": "evaluation", "split": self.phase, "stage": stage,
                     **adapter_statistics(model, features), "metrics": metrics(rows)})

    def finish(self, final_stage):
        if self.visited != list(range(1, final_stage + 1)):
            raise ValueError("refuse to summarize incomplete final evaluation")
        summary = {"split": self.phase, "final_stage": final_stage,
                   "provenance": self.args._provenance,
                   **summarize_rows(self.rows, final_stage)}
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"PGR diagnostic output: {self.directory}")
