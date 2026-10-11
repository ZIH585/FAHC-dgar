"""CPU-only, standard-library checks for mechanism-ranking statistics."""

import json
import math
import unittest

from src.fahc_mechanism_metrics import summarize_mechanism


class MechanismMetricsTest(unittest.TestCase):
    def test_average_tie_ranks(self):
        result = summarize_mechanism(
            [0, 1, 2], {0: 1, 1: 1, 2: 2},
            {0: 1, 1: 2, 2: 3}, {0: 1, 1: 2, 2: 3}, k=1)
        self.assertAlmostEqual(result["spearman_ab"], math.sqrt(3) / 2)
        self.assertEqual(result["vector_stats"]["risk_a"]["tied_item_count"], 2)
        self.assertEqual(result["vector_stats"]["risk_a"]["unique_count"], 2)

    def test_good_prediction_captures_oracle(self):
        scores = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
        result = summarize_mechanism([0, 1, 2, 3], scores, scores, scores, k=2)
        self.assertEqual(result["spearman_ab"], 1.0)
        self.assertEqual(result["spearman_mean_observed"], 1.0)
        self.assertEqual(result["topk_mean"], [3, 2])
        self.assertEqual(result["topk_overlap_fraction"], 1.0)
        self.assertEqual(result["topk_overlap_random_baseline"], 0.5)
        self.assertEqual(result["topk_jaccard"], 1.0)
        self.assertEqual(result["observed_topk_lift"], 1.0)
        self.assertEqual(result["oracle_capture_ratio"], 1.0)

    def test_stability_does_not_imply_prediction(self):
        scores = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
        observed = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
        result = summarize_mechanism([0, 1, 2, 3], scores, scores, observed, k=2)
        self.assertEqual(result["spearman_ab"], 1.0)
        self.assertEqual(result["spearman_mean_observed"], -1.0)
        self.assertEqual(result["observed_topk_lift"], -1.0)
        self.assertAlmostEqual(result["oracle_capture_ratio"], 0.2)

    def test_all_zero_ties_use_independent_reproducible_streams(self):
        ids = list(range(70))
        zeros = dict.fromkeys(ids, 0.0)
        first = summarize_mechanism(ids, zeros, zeros, zeros, k=35, seed=17)
        repeated = summarize_mechanism(ids[::-1], zeros, zeros, zeros, k=35, seed=17)
        self.assertEqual(first, repeated)
        self.assertNotEqual(set(first["topk_a"]), set(first["topk_b"]))
        self.assertLess(first["topk_overlap_fraction"], 1.0)
        self.assertEqual(first["topk_overlap_random_baseline"], 0.5)
        self.assertIsNone(first["spearman_ab"])
        self.assertIsNone(first["spearman_mean_observed"])
        self.assertIsNone(first["oracle_capture_ratio"])
        self.assertEqual(first["observed_topk_lift"], 0.0)
        self.assertEqual(first["vector_stats"]["risk_a"]["zero_fraction"], 1.0)
        self.assertEqual(first["vector_stats"]["risk_a"]["tied_item_fraction"], 1.0)
        json.dumps(first, allow_nan=False)

    def test_one_constant_vector_makes_correlations_undefined(self):
        ids = [0, 1, 2]
        constant = dict.fromkeys(ids, 1.0)
        varying = {0: 1.0, 1: 2.0, 2: 3.0}
        result = summarize_mechanism(ids, constant, varying, varying, k=1)
        self.assertIsNone(result["spearman_ab"])
        self.assertIsNone(result["spearman_a_observed"])
        self.assertEqual(result["spearman_b_observed"], 1.0)

    def test_empty_and_single_candidate(self):
        empty = summarize_mechanism([], {}, {}, {}, k=35)
        self.assertEqual(empty["effective_k"], 0)
        self.assertFalse(empty["selection_has_choice"])
        self.assertEqual(empty["topk_a"], [])
        self.assertIsNone(empty["topk_overlap_fraction"])
        self.assertIsNone(empty["topk_overlap_random_baseline"])
        self.assertIsNone(empty["observed_topk_lift"])
        self.assertIsNone(empty["vector_stats"]["risk_a"]["zero_fraction"])
        only = summarize_mechanism([5], {5: 1}, {5: 2}, {5: 3}, k=35)
        self.assertEqual(only["effective_k"], 1)
        self.assertFalse(only["selection_has_choice"])
        self.assertEqual(only["topk_overlap_fraction"], 1.0)
        self.assertEqual(only["topk_overlap_random_baseline"], 1.0)
        self.assertEqual(only["oracle_capture_ratio"], 1.0)
        self.assertIsNone(only["spearman_ab"])

    def test_large_finite_scores_do_not_overflow_derived_statistics(self):
        scores = {0: 1e308, 1: 1e308, 2: 1e307}
        result = summarize_mechanism([0, 1, 2], scores, scores, scores, k=2)
        self.assertTrue(math.isfinite(result["observed_mean_all"]))
        self.assertEqual(result["risk_mean"][0], 1e308)
        self.assertEqual(result["oracle_capture_ratio"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_equal_tiny_scores_keep_their_mean(self):
        tiny = math.nextafter(0.0, 1.0)
        result = summarize_mechanism([0], {0: tiny}, {0: tiny}, {0: tiny}, k=1)
        self.assertEqual(result["risk_mean"][0], tiny)
        self.assertEqual(result["oracle_capture_ratio"], 1.0)

    def test_invalid_scores_and_misaligned_ids_fail_fast(self):
        for value in (float("nan"), float("inf"), -0.1, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_mechanism([0], {0: value}, {0: 1}, {0: 1}, k=1)
        for mapping in ({}, {0: 1, 1: 2}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                summarize_mechanism([0], {0: 1}, mapping, {0: 1}, k=1)
        for ids in ([0, 0], [-1], [True], [1.5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                summarize_mechanism(ids, {}, {}, {}, k=1)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                summarize_mechanism([], {}, {}, {}, k=k)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                summarize_mechanism([], {}, {}, {}, k=1, seed=seed)


if __name__ == "__main__":
    unittest.main()
