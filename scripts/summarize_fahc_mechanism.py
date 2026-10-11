"""Export FAHC probe stability and observed-loss diagnostics, without test scores."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys


METRIC_COLUMNS = [
    "n_candidates", "effective_k", "selection_has_choice", "spearman_ab",
    "topk_overlap_fraction", "topk_overlap_random_baseline",
    "spearman_mean_observed", "observed_topk_lift", "oracle_capture_ratio",
    "spearman_a_observed", "spearman_b_observed", "observed_mean_all",
    "observed_mean_topk_mean", "configured_probe_spearman_observed",
    "configured_probe_observed_topk_lift",
]
REQUIRED_METRICS = METRIC_COLUMNS[:9]
STAT_COLUMNS = [f"{vector}_{field}"
                for vector in ("risk_a", "risk_b", "risk_mean", "observed")
                for field in ("zero_fraction", "tied_item_fraction", "unique_count")]
COLUMNS = [
    "run_id", "snapshot", "target", "complete_run", "batch_facts",
    "current_fact_count", "configured_probe_facts", "reference_steps",
    "actual_rgcn_steps", "candidate_count",
    "nonempty_candidate_count",
] + METRIC_COLUMNS + STAT_COLUMNS + ["source_file"]


def discover_logs(path: Path) -> list[Path]:
    """Find a supplied log, one run's log, or logs below an experiment root."""
    if not path.exists():
        raise ValueError(f"input does not exist: {path}")
    if path.is_file():
        return [path]
    direct = path / "logs" / "stages.jsonl"
    if direct.is_file():
        return [direct]
    # A logs directory itself is also convenient to inspect.
    direct = path / "stages.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(candidate for candidate in path.rglob("stages.jsonl")
                  if candidate.is_file() and candidate.parent.name == "logs")


def _integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _scalar(value, label: str):
    """Keep undefined statistics as None; reject values CSV cannot interpret."""
    if value is None or isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, boolean or null")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite or null")
    return value


def _records(path: Path) -> list[dict]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{index}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{index}: JSON record must be an object")
        records.append(record)
    return records


def read_log(path: Path) -> list[dict]:
    """Flatten only mechanism events; retain incomplete runs with a clear flag."""
    records = _records(path)
    mechanisms = [record for record in records if record.get("event") == "mechanism"]
    if not mechanisms:
        return []
    groups: dict[str, list[dict]] = {}
    for record in mechanisms:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{path}: mechanism event requires a nonempty run_id")
        _integer(record.get("snapshot"), f"{path}: snapshot")
        groups.setdefault(run_id, []).append(record)

    completion = {}
    for record in records:
        if record.get("event") != "mechanism_complete":
            continue
        run_id = record.get("run_id")
        if run_id is None and len(groups) == 1:
            run_id = next(iter(groups))
        if run_id not in groups:
            continue
        snapshots = record.get("snapshots")
        if not isinstance(snapshots, list):
            raise ValueError(f"{path}: mechanism_complete.snapshots must be a list")
        for snapshot in snapshots:
            _integer(snapshot, f"{path}: completed snapshot")
        # Use the most recent completion marker for an appended run.
        completion[run_id] = snapshots

    rows = []
    for run_id, stages in groups.items():
        snapshots = [record["snapshot"] for record in stages]
        expected = completion.get(run_id)
        complete = (expected is not None and len(expected) == len(set(expected))
                    and len(snapshots) == len(set(snapshots))
                    and set(expected) == set(snapshots))
        for record in stages:
            base = {
                "run_id": run_id, "snapshot": record["snapshot"],
                "complete_run": complete, "source_file": str(path.resolve()),
            }
            for field in ("batch_facts", "current_fact_count", "reference_steps",
                          "candidate_count", "nonempty_candidate_count"):
                base[field] = _integer(record.get(field), f"{path}: {field}")
            for field in ("configured_probe_facts", "actual_rgcn_steps"):
                value = record.get(field)
                base[field] = (_integer(value, f"{path}: {field}")
                               if value is not None else None)
            if base["nonempty_candidate_count"] > base["candidate_count"]:
                raise ValueError(f"{path}: nonempty candidate count exceeds total count")
            metrics = record.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError(f"{path}: mechanism.metrics must be a nonempty object")
            for target, values in sorted(metrics.items()):
                if not isinstance(values, dict):
                    raise ValueError(f"{path}: metrics[{target}] must be an object")
                missing = [field for field in REQUIRED_METRICS if field not in values]
                if missing:
                    raise ValueError(f"{path}: metrics[{target}] missing {', '.join(missing)}")
                n = _integer(values["n_candidates"], f"{path}: n_candidates")
                k = _integer(values["effective_k"], f"{path}: effective_k")
                if n != base["nonempty_candidate_count"] or k > n:
                    raise ValueError(f"{path}: metric candidate counts are inconsistent")
                if values["selection_has_choice"] is not (n > k):
                    raise ValueError(f"{path}: selection_has_choice is inconsistent")
                row = dict(base, target=target)
                row.update({field: _scalar(values.get(field), f"{path}: {field}")
                            for field in METRIC_COLUMNS})
                stats = values.get("vector_stats", {})
                if not isinstance(stats, dict):
                    raise ValueError(f"{path}: vector_stats must be an object")
                for vector in ("risk_a", "risk_b", "risk_mean", "observed"):
                    vector_stats = stats.get(vector, {})
                    if not isinstance(vector_stats, dict):
                        raise ValueError(f"{path}: vector_stats[{vector}] must be an object")
                    for field in ("zero_fraction", "tied_item_fraction", "unique_count"):
                        row[f"{vector}_{field}"] = _scalar(
                            vector_stats.get(field), f"{path}: {vector}.{field}")
                rows.append(row)
    return rows


def summarize_path(path: Path) -> list[dict]:
    rows = [row for log in discover_logs(path) for row in read_log(log)]
    if not rows:
        raise ValueError(f"no mechanism records found in: {path}")
    return rows


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="stages.jsonl, one run directory, or an experiment root")
    parser.add_argument("--csv", type=Path, help="also save the CSV to this path")
    args = parser.parse_args(argv)
    try:
        rows = summarize_path(args.input)
        output = csv_text(rows)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            args.csv.write_text(output, encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    sys.stdout.write(output)
    incomplete = sum(not row["complete_run"] for row in rows)
    print(f"Exported {len(rows)} stage/target rows; {incomplete} rows belong to "
          "incomplete runs. Blank cells mean undefined or unavailable, not zero. "
          "Compare overlap with its random baseline; selection_has_choice=False "
          "provides no selection evidence. Positive observed_topk_lift means "
          "the selected HCPs lost more than the candidate average in this loss "
          "diagnostic. These metrics do not establish held-out MRR improvement "
          "and do not rank configurations using TEST.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
