"""Summarize validation diagnostics; never use test MRR to rank configs."""

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import sys


def read_run(directory, include_test=False):
    config_path = directory / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    settings = config["settings"]
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    path = directory / "logs/stages.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()] if path.is_file() else []
    stages = [record for record in records if record.get("event") == "stage"]
    row = {"run": directory.name, "mode": settings["hcp_selection"],
           "seed": settings["seed"], "B": settings["fahc_fact_budget"],
           "eta": settings["fahc_eta"], "KD": settings["fahc_kd_weight"],
           "state": "finished" if status.get("python_exit_code") == 0
                    and status.get("tee_exit_code") == 0 else
                    ("failed" if status else "running_or_interrupted"),
           "wall_min": round(status["wall_seconds"] / 60, 2)
                       if "wall_seconds" in status else None}
    final_validations = [record for record in records
                        if record.get("event") == "final_validation"]
    if final_validations:
        full = final_validations[-1]
        full_past = {int(index): value for index, value in
                     full["per_snapshot_filtered_entity_mrr"].items()}
        row.update({
            "val_full_MRR": full["filtered_entity_mrr"],
            "val_full_past_MRR": full["past_macro_filtered_entity_mrr"],
            "val_full_current_MRR": full["current_filtered_entity_mrr"],
            "full_validation_complete": (
                set(full_past) == set(range(1, config["snapshot_count"]))),
        })
    if include_test:
        tests = [record for record in records if record.get("event") == "final_test"]
        if tests:
            final_test = tests[-1]
            row.update({
                "test_full_MRR": final_test["filtered_entity_mrr"],
                "test_full_past_MRR": final_test["past_macro_filtered_entity_mrr"],
                "test_full_current_MRR": final_test["current_filtered_entity_mrr"],
            })
    if not stages:
        return row
    final = max(stages, key=lambda record: record["snapshot"])
    arrivals = {record["snapshot"]: record["history_only_current_filtered_mrr"]
                for record in stages
                if record.get("history_only_current_filtered_mrr") is not None}
    past = {int(index): value for index, value in
            final.get("history_only_past_filtered_mrr", {}).items()}
    matched = sorted(set(arrivals) & set(past))
    differences = [past[index] - arrivals[index] for index in matched]
    row.update({
        "final_stage": final["snapshot"], "past_n": len(past),
        "past_complete": (set(past) == set(range(1, final["snapshot"]))
                          and final["snapshot"] == config["snapshot_count"] - 1
                          and set(arrivals) == set(range(1, final["snapshot"] + 1))),
        "val_past_MRR": statistics.mean(past.values()) if past else None,
        "val_current_MRR": arrivals.get(final["snapshot"]),
        "val_BWT": statistics.mean(differences) if differences else None,
        "val_arrival_forgetting_pos": statistics.mean(
            max(-difference, 0) for difference in differences) if differences else None,
    })
    controlled = [record for record in stages
                  if record.get("fact_budget_active")
                  and record.get("selected_nonempty_count", 0) > 0]
    if controlled:
        row.update({
            "replay_facts_median": statistics.median(
                record["selected_fact_count"] for record in controlled),
            "cap_reached_fraction": statistics.mean(
                record["selected_fact_count"] == settings["fahc_fact_budget"]
                for record in controlled),
        })
        fractions = [record["replay_retained_fraction"] for record in controlled
                     if record.get("replay_retained_fraction") is not None]
        if fractions:
            row["retained_fraction_median"] = statistics.median(fractions)
    gate = [record["kd_teacher_active_fact_count"] / record["selected_fact_count"]
            for record in stages if record.get("selected_fact_count", 0) > 0
            and "kd_teacher_active_fact_count" in record]
    if gate:
        row["KD_active_fraction_median"] = statistics.median(gate)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--include-test", action="store_true",
                        help="export test results only after freezing the configuration")
    args = parser.parse_args()
    rows = [row for directory in sorted(args.root.iterdir()) if directory.is_dir()
            and (row := read_run(directory, args.include_test)) is not None]
    columns = ["run", "state", "mode", "seed", "B", "eta", "KD",
               "val_full_past_MRR", "val_full_current_MRR", "val_full_MRR",
               "full_validation_complete",
               "val_past_MRR", "val_current_MRR", "val_BWT",
               "val_arrival_forgetting_pos", "past_n", "past_complete",
               "replay_facts_median", "retained_fraction_median",
               "cap_reached_fraction", "KD_active_fraction_median", "wall_min"]
    if args.include_test:
        columns.extend(["test_full_past_MRR", "test_full_current_MRR", "test_full_MRR"])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: round(value, 6) if isinstance(value, float) else value
                         for key, value in row.items()})
    sys.stdout.write(output.getvalue())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(output.getvalue())
    print("val_* columns use VALIDATION. val_full_* uses final DGAR inference; "
          "val_BWT uses history-only arrival vs final, not maximum-over-all-stages "
          "forgetting. Test columns are exported only with --include-test.", file=sys.stderr)


if __name__ == "__main__":
    main()
