"""Dependency-free checks for FAHC command-line and checkpoint configuration."""

import argparse
import ast
import copy
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.fahc_config import (ALGORITHM_VERSION, experiment_settings,
                             experiment_signature, parse_bool, validate_args)


def sample_args(**overrides):
    """Representative defaults from src/main.py, without importing PyTorch."""
    values = dict(
        dataset="ICEWS14s", seed=1, gpu=0, test=False,
        grid_search=False, tune="n_hidden,n_layers", run_analysis=False,
        fahc_log=None, fahc_diagnostics=False,
        fahc_eval_every=1, fahc_past_eval_samples=4,
        hcp_selection="risk+coverage", reply_batch=35,
        fahc_candidate_multiplier=2.0, fahc_fact_budget=128,
        fahc_eta=1.0, fahc_entity_weight=1.0,
        fahc_time_bin_size=10, fahc_probe_steps=1,
        fahc_probe_facts=128, fahc_probe_lr=None,
        fahc_score_batch_size=256, fahc_kd_weight=0.1,
        fahc_kd_temperature=2.0, fahc_kd_confidence=0.5,
        delete_feature_reply=False, delete_his_prompt=False,
        add_history=True, diffuc="add",
        n_epochs=1, regcn_epochs=10, diffu_epochs=10,
        n_layers=2, train_history_len=3, test_history_len=3,
        dilate_len=1, n_hidden=200, hidden_size=200,
        attn_heads=2, heads=4, num_blocks=2, num_blocks_cross=0,
        diffusion_steps=35, max_len=64, his_max_len=128,
        batch_size=1, accumulation_steps=2, start_reply=0,
        reply_batch_num=128, lr=0.001, grad_norm=1.0,
        mu=1.0, mu_r=1.0, weight=0.5, task_weight=0.7,
        discount=1.0, angle=10.0, classifier_scale=1.0,
        lambda_uncertainty=0.01, pattern_noise_radio=1.0,
        dropout=0.2, input_dropout=0.2, hidden_dropout=0.2,
        feat_dropout=0.2, emb_dropout=0.2,
        temperature_object=0.5, scale=50.0,
        beta_start=0.0001, beta_end=0.02,
        encoder="uvrgcn", decoder="convtranse", n_bases=100,
        n_basis=100, add_static_graph=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FAHCConfigTests(unittest.TestCase):
    def test_parse_bool_accepts_common_values_without_string_truthiness(self):
        for value in (True, 1, "TRUE", " yes ", "On", "t", "y"):
            with self.subTest(value=value):
                self.assertIs(parse_bool(value), True)
        for value in (False, 0, "FALSE", " no ", "Off", "f", "n"):
            with self.subTest(value=value):
                self.assertIs(parse_bool(value), False)
        for value in (2, "maybe", None, object()):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_bool(value)

    def test_valid_configuration_and_legacy_false_string(self):
        args = sample_args(delete_feature_reply="false", delete_his_prompt="no",
                           add_history="yes", diffuc="True")
        validate_args(args)
        self.assertIs(args.delete_feature_reply, False)
        self.assertIs(args.delete_his_prompt, False)
        self.assertIs(args.add_history, True)
        self.assertEqual(args.diffuc, "add")

        linear = sample_args(diffuc="Linear")
        validate_args(linear)
        self.assertEqual(linear.diffuc, "Linear")

    def test_rejects_missing_replay_prerequisites_and_unmatched_kd(self):
        cases = (
            (dict(diffuc=False), "diffusion feature replay"),
            (dict(diffuc="false"), "diffusion feature replay"),
            (dict(delete_feature_reply=True), "diffusion feature replay"),
            (dict(delete_his_prompt=True), "historical HCP prompts"),
            (dict(add_history=False), "historical context"),
            (dict(train_history_len=0), "train-history-len"),
            (dict(n_epochs=2), "chronological stages"),
            (dict(reply_batch=0), "fact budget"),
            (dict(reply_batch=35, fahc_fact_budget=34), "fact budget"),
            (dict(hcp_selection="original", fahc_kd_weight=0.1), "budget-matched"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(sample_args(**overrides))

    def test_rejects_invalid_numeric_parameters(self):
        cases = (
            ("fahc_candidate_multiplier", float("nan")),
            ("fahc_eta", float("inf")),
            ("fahc_entity_weight", -0.1),
            ("fahc_kd_weight", -0.1),
            ("fahc_kd_temperature", 0.0),
            ("fahc_kd_confidence", 1.1),
            ("fahc_probe_lr", float("nan")),
            ("fahc_probe_steps", 0),
            ("fahc_score_batch_size", 0),
            ("fahc_eval_every", -1),
            ("lr", 0.0),
            ("mu", -1.0),
            ("mu_r", "0.5"),
            ("task_weight", 1.1),
            ("dropout", 1.1),
            ("regcn_epochs", 0),
            ("seed", 2**32),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    validate_args(sample_args(**{name: value}))

    def test_rejects_incompatible_model_and_schedule_settings(self):
        cases = (
            dict(n_hidden=128),
            dict(n_hidden=201, hidden_size=201),
            dict(heads=3),
            dict(start_reply=10),
            dict(diffusion_steps=1),
            dict(discount=0.5),
            dict(beta_start=0.03, beta_end=0.02),
            dict(scale=51.0, noise_schedule="linear"),
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validate_args(sample_args(**overrides))

    def test_original_mode_does_not_require_fahc_replay(self):
        args = sample_args(hcp_selection="original", fahc_kd_weight=0,
                           diffuc=False, add_history=False,
                           delete_feature_reply=True,
                           delete_his_prompt=True, start_reply=10,
                           n_epochs=2)
        validate_args(args)
        self.assertIs(args.diffuc, False)
        with self.assertRaisesRegex(ValueError, "chronological stages"):
            validate_args(sample_args(hcp_selection="original",
                                      fahc_kd_weight=0,
                                      fahc_diagnostics=True, n_epochs=2))

    def test_signature_is_stable_and_excludes_runtime_options(self):
        args = sample_args()
        validate_args(args)
        signature = experiment_signature(args)
        self.assertRegex(signature, re.compile(r"^[0-9a-f]{10}$"))
        self.assertEqual(experiment_settings(args)["algorithm_version"],
                         ALGORITHM_VERSION)
        reordered = SimpleNamespace(**dict(reversed(list(vars(args).items()))))
        self.assertEqual(experiment_signature(reordered), signature)

        runtime = copy.deepcopy(args)
        runtime.gpu = 7
        runtime.test = True
        runtime.grid_search = True
        runtime.tune = "different"
        runtime.fahc_eval_every = 3
        runtime.fahc_past_eval_samples = 0
        runtime.fahc_diagnostics = True
        runtime.fahc_log = "/tmp/other.jsonl"
        runtime.test_history_len = 9
        runtime.vl = True
        runtime.num_nodes = 1000
        runtime.num_rels = 10
        self.assertEqual(experiment_signature(runtime), signature)

    def test_signature_changes_with_training_or_model_settings(self):
        baseline = sample_args()
        base_signature = experiment_signature(baseline)
        for name, value in (("seed", 2), ("mu", 0.5), ("n_hidden", 256),
                            ("fahc_fact_budget", 256), ("fahc_eta", 2.0),
                            ("encoder", "another"), ("regcn_epochs", 12)):
            with self.subTest(name=name):
                changed = copy.deepcopy(baseline)
                setattr(changed, name, value)
                self.assertNotEqual(experiment_signature(changed), base_signature)

        effective_lr = sample_args(fahc_probe_lr=0.001)
        self.assertEqual(experiment_signature(effective_lr), base_signature)

    def test_actual_main_parser_parses_numeric_and_boolean_flags(self):
        # Importing main requires PyTorch and DGL. Extract only its parser
        # function, which has no model dependencies, so this remains CPU-only.
        main_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
        tree = ast.parse(main_path.read_text(encoding="utf-8"),
                         filename=str(main_path))
        parser_function = next(node for node in tree.body
                               if isinstance(node, ast.FunctionDef)
                               and node.name == "build_arg_parser")
        module = ast.Module(body=[parser_function], type_ignores=[])
        namespace = {"argparse": argparse, "parse_bool": parse_bool}
        exec(compile(module, str(main_path), "exec"), namespace)
        parser = namespace["build_arg_parser"]()

        args = parser.parse_args([
            "--hcp-selection", "risk", "--mu", "0.5", "--mu_r", "0.25",
            "--diffuc", "True", "--delete_feature_reply", "false",
            "--delete_his_prompt", "false", "--add-static-graph", "false",
        ])
        self.assertEqual((args.mu, args.mu_r), (0.5, 0.25))
        self.assertIs(args.delete_feature_reply, False)
        self.assertIs(args.delete_his_prompt, False)
        self.assertIs(args.add_static_graph, False)
        validate_args(args)
        self.assertEqual(args.diffuc, "add")

        off = parser.parse_args(["--diffuc", "False"])
        validate_args(off)
        self.assertIs(off.diffuc, False)


if __name__ == "__main__":
    unittest.main()
