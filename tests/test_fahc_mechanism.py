"""Diagnostic state checks on CPU; optimizer steps are always mocked."""

import copy
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from diffusion.fahc_selector import HCPCandidate
from src.fahc_mechanism import (begin_mechanism_diagnostic,
                                 finish_mechanism_diagnostic)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2, 4)

    def get_entity_context(self, *args):
        self.h = self.linear.weight + 0
        return self.linear.weight, None

    def get_entity_logits(self, history, triples, static, use_cuda, context=None):
        return self.linear(self.bn(triples[:, :2].float()))

    def get_loss(self, history, triples, static, use_cuda):
        random.random()
        np.random.random()
        logits = self.get_entity_logits(history, triples, static, use_cuda)
        loss = F.cross_entropy(logits, triples[:, 2])
        return loss, loss * 0, loss * 0, logits, loss * 0


class MechanismStateTest(unittest.TestCase):
    def setUp(self):
        torch.default_generator.manual_seed(5)
        self.model = ToyModel()
        self.teacher = copy.deepcopy(self.model).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.current = torch.tensor([[i % 4, i % 2, (i + 1) % 4]
                                     for i in range(8)], dtype=torch.long)
        self.candidates = [HCPCandidate(0, ((0, 0, 1), (1, 1, 2))),
                           HCPCandidate(1, ((2, 0, 3),)), HCPCandidate(2, ())]
        self.args = SimpleNamespace(seed=1, fahc_probe_facts=128,
                                    fahc_probe_steps=1, fahc_probe_lr=None,
                                    lr=0.01, task_weight=0.7, grad_norm=1,
                                    fahc_score_batch_size=2,
                                    fahc_mechanism_reference_steps=3, reply_batch=1)

    def begin(self):
        return begin_mechanism_diagnostic(
            self.args, 60, self.model, self.teacher, self.candidates,
            self.current, [], None, False, self.optimizer)

    def test_disjoint_probes_bn_isolation_and_complete_restoration(self):
        self.model.train()
        self.model.bn.eval()
        self.model.h = torch.tensor([11.0])
        cache = self.model.h
        weights = copy.deepcopy(self.model.state_dict())
        modes = [module.training for module in self.model.modules()]
        teacher_weights = copy.deepcopy(self.teacher.state_dict())
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        with (patch.object(torch.optim.Adam, "step", return_value=None) as step,
              patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA touched")),
              patch.object(torch.cuda, "set_rng_state", side_effect=AssertionError("CUDA touched"))):
            pending = self.begin()
        self.assertEqual(step.call_count, 6)
        record = pending["record"]
        self.assertEqual(record["batch_facts"], 4)
        self.assertEqual(record["configured_probe_facts"], 8)
        self.assertFalse(set(record["batch_a_positions"]) & set(record["batch_b_positions"]))
        self.assertEqual(record["metrics"]["current_only"]["n_candidates"], 2)
        self.assertEqual(set(record["metrics"]["current_only"]["risk_mean"]), {0, 1})
        self.assertTrue(all(value == 0 for value in record["risk_a_fixed_bn"].values()))
        self.assertTrue(all(value == 0 for value in record["risk_b_fixed_bn"].values()))
        for name, value in weights.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]), name)
        for name, value in teacher_weights.items():
            self.assertTrue(torch.equal(value, self.teacher.state_dict()[name]), name)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertIs(self.model.h, cache)
        self.assertEqual(python_rng, random.getstate())
        restored_np = np.random.get_state()
        self.assertTrue(np.array_equal(numpy_rng[1], restored_np[1]))
        self.assertEqual(numpy_rng[2:], restored_np[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(self.optimizer.state_dict()["state"], {})

        final = finish_mechanism_diagnostic(self.args, self.model, pending, [], None, False)
        self.assertEqual(set(final["metrics"]),
                         {"current_only", "fixed_bn_current_only", "dgar_history_only"})
        self.assertTrue(all(value == 0 for value in final["observed_dgar_history_only"].values()))
        self.assertIs(self.model.h, cache)
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_empty_hcps_are_reported_without_fake_ranking_evidence(self):
        self.candidates = [HCPCandidate(0, ()), HCPCandidate(1, ())]
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("empty pool updated")):
            pending = self.begin()
        self.assertEqual(pending["record"]["nonempty_candidate_count"], 0)
        metric = pending["record"]["metrics"]["current_only"]
        self.assertIsNone(metric["spearman_ab"])
        self.assertFalse(metric["selection_has_choice"])


if __name__ == "__main__":
    unittest.main()
