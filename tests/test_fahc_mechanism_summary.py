"""Standard-library checks for mechanism export and incomplete-run reporting."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_fahc_mechanism import main, summarize_path, csv_text
from src.fahc_mechanism_metrics import summarize_mechanism


def mechanism(snapshot=100, run_id="diagnostic", targets=None):
    values = summarize_mechanism(
        [0, 1, 2, 3], {0: 0, 1: 1, 2: 2, 3: 3},
        {0: 0, 1: 1, 2: 2, 3: 3}, {0: 3, 1: 2, 2: 1, 3: 0}, k=2)
    return {
        "event": "mechanism", "run_id": run_id, "snapshot": snapshot,
        "batch_facts": 128, "current_fact_count": 400, "reference_steps": 10,
        "candidate_count": 5, "nonempty_candidate_count": 4,
        "metrics": targets or {"current_only": values},
    }


def write_log(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


class MechanismSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_file_run_directory_and_root_discovery(self):
        path = write_log(self.root / "nested" / "run" / "logs" / "stages.jsonl", [
            {"event": "stage", "snapshot": 99, "test_MRR": 0.9},
            mechanism(), {"event": "mechanism_complete", "snapshots": [100]},
        ])
        by_file = summarize_path(path)
        self.assertEqual(by_file, summarize_path(path.parent.parent))
        self.assertEqual(by_file, summarize_path(self.root))
        row = by_file[0]
        self.assertTrue(row["complete_run"])
        self.assertEqual(row["n_candidates"], 4)
        self.assertEqual(row["candidate_count"], 5)
        self.assertEqual(row["effective_k"], 2)
        self.assertEqual(row["spearman_ab"], 1)
        self.assertEqual(row["spearman_mean_observed"], -1)
        self.assertEqual(row["observed_topk_lift"], -1)
        self.assertNotIn("test_MRR", row)

    def test_multiple_targets_keep_distinct_results_and_nulls(self):
        regular = mechanism()["metrics"]["current_only"]
        zeros = dict.fromkeys(range(4), 0.0)
        undefined = summarize_mechanism(list(range(4)), zeros, zeros, zeros, k=2)
        path = write_log(self.root / "stages.jsonl", [mechanism(targets={
            "current_only": regular, "dgar_history_only": undefined,
            "fixed_bn_current_only": regular,
        })])
        rows = summarize_path(path)
        self.assertEqual(len(rows), 3)
        exported = {row["target"]: row for row in csv.DictReader(io.StringIO(csv_text(rows)))}
        self.assertEqual(exported["current_only"]["spearman_mean_observed"], "-1.0")
        self.assertEqual(exported["dgar_history_only"]["spearman_mean_observed"], "")
        self.assertEqual(exported["dgar_history_only"]["oracle_capture_ratio"], "")
        self.assertEqual(exported["dgar_history_only"]["observed_topk_lift"], "0.0")
        self.assertEqual(exported["dgar_history_only"]["risk_a_zero_fraction"], "1.0")

    def test_configured_single_probe_and_actual_steps_are_optional(self):
        record = mechanism()
        path = write_log(self.root / "stages.jsonl", [record])
        legacy = summarize_path(path)[0]
        for field in ("configured_probe_facts", "actual_rgcn_steps",
                      "configured_probe_spearman_observed",
                      "configured_probe_observed_topk_lift"):
            self.assertIsNone(legacy[field])
        record.update(configured_probe_facts=128, actual_rgcn_steps=6)
        record["metrics"]["current_only"].update(
            configured_probe_spearman_observed=0.75,
            configured_probe_observed_topk_lift=0.03)
        write_log(path, [record])
        exported = next(csv.DictReader(io.StringIO(csv_text(summarize_path(path)))))
        self.assertEqual(exported["configured_probe_facts"], "128")
        self.assertEqual(exported["actual_rgcn_steps"], "6")
        self.assertEqual(exported["configured_probe_spearman_observed"], "0.75")
        self.assertEqual(exported["configured_probe_observed_topk_lift"], "0.03")

    def test_incomplete_or_mismatched_completion_is_flagged(self):
        for completion in (None, [100, 101], [101], [100, 100]):
            with self.subTest(completion=completion):
                records = [mechanism()]
                if completion is not None:
                    records.append({"event": "mechanism_complete", "snapshots": completion})
                path = write_log(self.root / "stages.jsonl", records)
                self.assertFalse(summarize_path(path)[0]["complete_run"])

    def test_complete_multiple_stages_and_duplicate_stage(self):
        records = [mechanism(100), mechanism(200),
                   {"event": "mechanism_complete", "snapshots": [200, 100]}]
        path = write_log(self.root / "stages.jsonl", records)
        self.assertTrue(all(row["complete_run"] for row in summarize_path(path)))
        records.insert(0, mechanism(100))
        write_log(path, records)
        self.assertTrue(all(not row["complete_run"] for row in summarize_path(path)))

    def test_complete_marker_does_not_complete_another_run(self):
        path = write_log(self.root / "stages.jsonl", [
            mechanism(run_id="a"), mechanism(run_id="b"),
            {"event": "mechanism_complete", "run_id": "a", "snapshots": [100]},
        ])
        complete = {row["run_id"]: row["complete_run"] for row in summarize_path(path)}
        self.assertEqual(complete, {"a": True, "b": False})

    def test_recursive_root_ignores_runs_without_mechanism_records(self):
        write_log(self.root / "ordinary" / "logs" / "stages.jsonl", [{"event": "stage"}])
        write_log(self.root / "partial" / "logs" / "stages.jsonl", [mechanism()])
        write_log(self.root / "complete" / "logs" / "stages.jsonl", [
            mechanism(run_id="finished"),
            {"event": "mechanism_complete", "run_id": "finished", "snapshots": [100]},
        ])
        rows = summarize_path(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["complete_run"] for row in rows), 1)

    def test_missing_input_no_mechanisms_and_invalid_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            summarize_path(self.root / "missing")
        path = write_log(self.root / "stages.jsonl", [{"event": "final_test"}])
        with self.assertRaisesRegex(ValueError, "no mechanism records"):
            summarize_path(path)
        invalid = mechanism()
        invalid["nonempty_candidate_count"] = 3
        write_log(path, [invalid])
        with self.assertRaisesRegex(ValueError, "counts are inconsistent"):
            summarize_path(path)

    def test_cli_exports_file_and_error_is_explicit(self):
        path = write_log(self.root / "stages.jsonl", [mechanism()])
        output_path = self.root / "export" / "mechanism.csv"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main([str(path), "--csv", str(output_path)]), 0)
        self.assertEqual(output_path.read_text(), stdout.getvalue().replace("\r\n", "\n"))
        self.assertIn("Blank cells mean undefined", stderr.getvalue())
        self.assertIn("1 rows belong to incomplete runs", stderr.getvalue())
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            main([str(self.root / "missing")])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("input does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
