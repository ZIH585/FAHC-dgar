"""Launcher and metric checks with a fake trainer; never run CUDA or training."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data1/zihao/miniconda3/envs/dgar_env/bin/python"
spec = importlib.util.spec_from_file_location(
    "fahc_summary", ROOT / "scripts/summarize_fahc_runs.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ExperimentToolsTest(unittest.TestCase):
    def launcher(self, root, fake_python=None, dry=False):
        env = os.environ.copy()
        env.update(DGAR_RUN_ROOT=str(root / "runs"),
                   DGAR_PYTHON=str(fake_python or PYTHON),
                   PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        command = ["bash", str(ROOT / "scripts/run_fahc_experiment.sh")]
        if dry:
            command.append("--dry-run")
        command.extend(["test", "0", "ICEWS14s", "1", "random", "4096", "0.1", "0"])
        return subprocess.run(command, env=env, capture_output=True, text=True)

    def fake_python(self, root, exit_code=0):
        path = root / "fake-python"
        path.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ $1 == -c ]]; then exit 0; fi\n'  # Mock CUDA check.
            'if [[ $1 == -u ]]; then\n'
            '  printf "simulated trainer, cwd=%s\\n" "$PWD"\n'
            '  printf "isolated checkpoint\\n" > ../models/fake.txt\n'
            '  printf "isolated diffusion\\n" > ../diffu_models/fake.txt\n'
            f'  exit {exit_code}\n'
            'fi\n'
            f'exec {PYTHON} "$@"\n')
        path.chmod(0o755)
        return path

    def test_dry_run_creates_no_output_and_imports_no_cuda_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.launcher(root, dry=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "runs").exists())
            self.assertIn("--fahc-past-eval-samples 0", result.stdout)

    def test_checkpoints_logs_and_data_link_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.launcher(root, self.fake_python(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            run = next(path for path in (root / "runs").iterdir()
                       if path.name != ".locks")
            self.assertEqual((run / "data").resolve(), ROOT / "data")
            self.assertTrue((run / "models/fake.txt").is_file())
            self.assertTrue((run / "diffu_models/fake.txt").is_file())
            self.assertIn(str(run / "src"), (run / "logs/train.log").read_text())
            self.assertEqual(json.loads((run / "status.json").read_text())["python_exit_code"], 0)

    def test_trainer_failure_is_logged_and_pipeline_returns_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.launcher(root, self.fake_python(root, exit_code=7))
            self.assertNotEqual(result.returncode, 0)
            run = next(path for path in (root / "runs").iterdir()
                       if path.name != ".locks")
            self.assertEqual(json.loads((run / "status.json").read_text())["python_exit_code"], 7)
            self.assertTrue((run / "logs/train.log").is_file())

    def test_summary_distinguishes_signed_bwt_and_clipped_arrival_forgetting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            settings = dict(hcp_selection="random", seed=1, fahc_fact_budget=128,
                            fahc_eta=0.1, fahc_kd_weight=0)
            (root / "config.json").write_text(json.dumps(
                {"settings": settings, "snapshot_count": 4}))
            records = [dict(event="stage", snapshot=index,
                            history_only_current_filtered_mrr=value,
                            fact_budget_active=True, selected_nonempty_count=1,
                            selected_fact_count=128, replay_retained_fraction=0.5)
                       for index, value in ((1, 0.4), (2, 0.3), (3, 0.5))]
            records[-1]["history_only_past_filtered_mrr"] = {"1": 0.3, "2": 0.5}
            records.append(dict(
                event="final_validation", filtered_entity_mrr=0.7,
                past_macro_filtered_entity_mrr=0.6,
                current_filtered_entity_mrr=0.9,
                per_snapshot_filtered_entity_mrr={"1": 0.5, "2": 0.7, "3": 0.9}))
            records.insert(0, dict(
                event="final_test", filtered_entity_mrr=0.99,
                past_macro_filtered_entity_mrr=0.98,
                current_filtered_entity_mrr=1.0))
            (root / "logs/stages.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records))
            row = summary.read_run(root)
            self.assertAlmostEqual(row["val_BWT"], 0.05)
            self.assertAlmostEqual(row["val_arrival_forgetting_pos"], 0.05)
            self.assertAlmostEqual(row["val_past_MRR"], 0.4)
            self.assertTrue(row["past_complete"])
            self.assertEqual(row["val_full_past_MRR"], 0.6)
            self.assertEqual(row["val_full_current_MRR"], 0.9)
            self.assertTrue(row["full_validation_complete"])
            self.assertNotIn("test_full_MRR", row)
            self.assertEqual(summary.read_run(root, include_test=True)["test_full_MRR"], 0.99)
            records[-2]["history_only_past_filtered_mrr"].pop("2")
            records[-1]["per_snapshot_filtered_entity_mrr"].pop("2")
            (root / "logs/stages.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records))
            self.assertFalse(summary.read_run(root)["past_complete"])
            self.assertFalse(summary.read_run(root)["full_validation_complete"])


if __name__ == "__main__":
    unittest.main()
