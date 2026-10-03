import pickle
import random
import tempfile
import unittest
from pathlib import Path

from diffusion.fahc_selector import (
    HCPCandidate,
    eligible_history_indices,
    load_hcp_candidates,
    materialize_replay,
    positive_loss_increase,
    recent_candidate_indices,
    sample_candidate_indices,
    sample_scoring_candidates,
    select_hcp,
)


class FAHCSelectorTests(unittest.TestCase):
    def test_historical_range_and_candidate_pool(self):
        self.assertEqual(eligible_history_indices(2), [0])
        self.assertEqual(eligible_history_indices(5), [0, 1, 2, 3])
        self.assertEqual(recent_candidate_indices(5, 2), [2, 3])
        pool = sample_candidate_indices(20, 3, multiplier=2, rng=random.Random(7))
        self.assertEqual(len(pool), 6)
        self.assertEqual(pool, sorted(set(pool)))
        self.assertTrue(set(pool).issubset(set(range(19))))

    def test_load_existing_hcp_format_by_current_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            history = {
                "1": {(1, 2, 3), (1, 4, 5)},
                "9": {(9, 2, 1)},
            }
            with (Path(directory) / "snap_0.pkl").open("wb") as stream:
                pickle.dump(history, stream)
            candidates = load_hcp_candidates(
                "unused", [0], [1], history_dir=Path(directory)
            )
        self.assertEqual(candidates, [HCPCandidate(0, ((1, 2, 3), (1, 4, 5)))])

    def test_risk_clips_each_fact_before_mean(self):
        self.assertEqual(positive_loss_increase([2.0, 1.0], [1.0, 3.0]), 1.0)
        self.assertEqual(positive_loss_increase([], []), 0.0)
        with self.assertRaises(ValueError):
            positive_loss_increase([1.0], [])

    def test_coverage_changes_greedy_choice_and_respects_k(self):
        candidates = [
            HCPCandidate(0, ((1, 1, 2),)),
            HCPCandidate(1, ((1, 1, 2),)),
            HCPCandidate(11, ((3, 2, 4),)),
        ]
        risks = {0: 1.0, 1: 0.9, 11: 0.1}
        by_risk = select_hcp(candidates, 2, risks, mode="risk")
        with_coverage = select_hcp(
            candidates, 2, risks, mode="risk+coverage", eta=2,
            time_bin_size=10,
        )
        self.assertEqual(by_risk.indices, (0, 1))
        self.assertEqual(with_coverage.indices, (0, 11))
        self.assertEqual(select_hcp(candidates, 2, mode="recent").indices, (11, 1))

    def test_inverse_relations_share_coverage_when_modulus_is_given(self):
        candidates = [HCPCandidate(0, ((0, 0, 1),)),
                      HCPCandidate(1, ((0, 2, 1),)),
                      HCPCandidate(2, ((0, 1, 1),))]
        raw = select_hcp(candidates, 2, mode="coverage", entity_weight=0)
        canonical = select_hcp(candidates, 2, mode="coverage",
                               entity_weight=0, relation_modulus=2)
        self.assertEqual(raw.indices, (0, 1))
        self.assertEqual(canonical.indices, (0, 2))

    def test_fact_cap_deduplicates_and_round_robins(self):
        candidates = [
            HCPCandidate(0, ((1, 1, 2), (2, 1, 3))),
            HCPCandidate(1, ((1, 1, 2), (4, 2, 5), (6, 2, 7))),
        ]
        result = select_hcp(candidates, 2, mode="recent", fact_budget=3)
        self.assertEqual(len(result.indices), 2)
        self.assertEqual(result.triples, ((1, 1, 2), (2, 1, 3), (4, 2, 5)))
        self.assertEqual(len(set(result.triples)), 3)
        self.assertEqual(result.nonempty_count, 2)
        self.assertEqual(result.fact_count, 3)

    def test_budget_preserves_every_scored_fact_and_fills_remaining_slots(self):
        full = [
            HCPCandidate(0, tuple((0, 0, object_id) for object_id in range(5))),
            HCPCandidate(1, tuple((1, 1, object_id) for object_id in range(5))),
        ]
        scored = sample_scoring_candidates(full, k=2, fact_budget=5, seed=11)
        self.assertEqual([len(candidate.triples) for candidate in scored], [2, 2])
        chosen = select_hcp(scored, k=2, mode="recent")
        replay = materialize_replay(chosen, full, fact_budget=5, seed=11)
        self.assertEqual(replay.fact_count, 5)
        self.assertTrue(set(chosen.triples).issubset(set(replay.triples)))
        self.assertEqual(replay.indices, chosen.indices)
        self.assertEqual(
            replay.triples,
            materialize_replay(chosen, full, fact_budget=5, seed=11).triples)
        with self.assertRaises(ValueError):
            sample_scoring_candidates(full, k=2, fact_budget=1, seed=11)

    def test_greedy_fills_empty_history_only_after_nonempty(self):
        candidates = [
            HCPCandidate(0, ()),
            HCPCandidate(1, ((2, 1, 3),)),
            HCPCandidate(2, ()),
        ]
        result = select_hcp(candidates, 2, {1: 0.0}, mode="risk")
        self.assertEqual(result.indices, (1, 0))
        self.assertEqual(result.nonempty_count, 1)
        self.assertEqual(result.fact_count, 1)


if __name__ == "__main__":
    unittest.main()
