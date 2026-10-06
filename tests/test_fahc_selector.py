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
        raw = select_hcp(candidates, 2, mode="coverage", entity_weight=0,
                         rng=random.Random(1))
        canonical = select_hcp(candidates, 2, mode="coverage",
                               entity_weight=0, relation_modulus=2,
                               rng=random.Random(1))
        self.assertEqual(raw.indices, (0, 1))
        self.assertEqual(canonical.indices, (0, 2))

    def test_fact_cap_deduplicates_full_candidates(self):
        candidates = [
            HCPCandidate(0, ((1, 1, 2), (2, 1, 3))),
            HCPCandidate(1, ((1, 1, 2), (4, 2, 5), (6, 2, 7))),
        ]
        selected = select_hcp(candidates, 2, mode="recent")
        result = materialize_replay(selected, candidates, fact_budget=3, seed=7)
        self.assertEqual(len(result.indices), 2)
        self.assertEqual(len(set(result.triples)), 3)
        self.assertTrue(set(result.triples).issubset(set(selected.triples)))
        self.assertEqual(result.triples,
                         materialize_replay(selected, candidates, 3, 7).triples)
        self.assertEqual(result.nonempty_count, 2)
        self.assertEqual(result.fact_count, 3)

    def test_round_robin_gives_each_slice_a_share_of_budget(self):
        candidates = [
            HCPCandidate(0, tuple((0, 0, object_id) for object_id in range(10))),
            HCPCandidate(1, tuple((1, 1, object_id) for object_id in range(10))),
        ]
        chosen = select_hcp(candidates, k=2, mode="recent")
        replay = materialize_replay(chosen, candidates, fact_budget=5, seed=11)
        counts = [sum(subject == index for subject, _, _ in replay.triples)
                  for index in (0, 1)]
        self.assertEqual(sorted(counts), [2, 3])

    def test_duplicate_slices_can_underfill_the_budget(self):
        candidates = [HCPCandidate(0, ((1, 0, 2),)),
                      HCPCandidate(1, ((1, 0, 2),)),
                      HCPCandidate(2, ())]
        chosen = select_hcp(candidates, k=3, mode="recent")
        replay = materialize_replay(chosen, candidates, fact_budget=8, seed=5)
        self.assertEqual(len(replay.indices), 3)
        self.assertEqual(replay.nonempty_count, 2)
        self.assertEqual(replay.fact_count, 1)

    def test_full_hcp_can_be_scored_before_smaller_replay_cap(self):
        full = [
            HCPCandidate(0, tuple((0, 0, object_id) for object_id in range(5))),
            HCPCandidate(1, tuple((1, 1, object_id) for object_id in range(5))),
        ]
        chosen = select_hcp(full, k=2, mode="recent")
        self.assertEqual(chosen.fact_count, 10)
        replay = materialize_replay(chosen, full, fact_budget=5, seed=11)
        self.assertEqual(replay.fact_count, 5)
        self.assertTrue(set(replay.triples).issubset(set(chosen.triples)))
        self.assertEqual(replay.indices, chosen.indices)
        self.assertEqual(
            replay.triples,
            materialize_replay(chosen, full, fact_budget=5, seed=11).triples)
        # The deprecated quota helper is still available for explicit
        # scoring-cost experiments, but it is not used by main training.
        scored = sample_scoring_candidates(full, k=2, fact_budget=5, seed=11)
        self.assertEqual([len(candidate.triples) for candidate in scored], [2, 2])
        with self.assertRaises(ValueError):
            sample_scoring_candidates(full, k=2, fact_budget=1, seed=11)

    def test_greedy_fills_empty_history_only_after_nonempty(self):
        candidates = [
            HCPCandidate(0, ()),
            HCPCandidate(1, ((2, 1, 3),)),
            HCPCandidate(2, ()),
        ]
        result = select_hcp(candidates, 2, {1: 0.0}, mode="risk")
        self.assertEqual(result.indices[0], 1)
        self.assertIn(result.indices[1], (0, 2))
        self.assertEqual(result.nonempty_count, 1)
        self.assertEqual(result.fact_count, 1)

    def test_equal_risks_have_seeded_unbiased_ties(self):
        candidates = [HCPCandidate(i, ((i, 0, i + 1),)) for i in range(6)]
        risks = {i: 0.0 for i in range(6)}
        choices = [select_hcp(candidates, 2, risks, mode="risk",
                              rng=random.Random(seed)).indices
                   for seed in range(12)]
        self.assertGreater(len(set(choices)), 1)
        self.assertEqual(choices[3], select_hcp(
            candidates, 2, risks, mode="risk", rng=random.Random(3)).indices)
        self.assertEqual(select_hcp(candidates, 2, risks, mode="risk").indices,
                         select_hcp(candidates, 2, risks, mode="risk").indices)


if __name__ == "__main__":
    unittest.main()
