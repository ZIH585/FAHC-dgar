import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from diffusion.fahc_selector import HCPCandidate
from diffusion.model_21 import Att_Diffuse_model
from rgcn.utils import build_sub_graph
from src.fahc import probe_hcp_risks, replay_kd_loss, teacher_replay_logits
from src.rrgcn import RecurrentRGCN


class ProbeIntegrationTest(unittest.TestCase):
    def test_feature_generator_uses_selected_hcp_facts(self):
        class FeatureStub:
            def __init__(self):
                self.seen = None

            def __call__(self, sequence, tag, *args, **kwargs):
                self.seen = tag
                return (None, torch.ones((len(tag), 3, 8)), None, None,
                        None, None, None, None, None)

        stub = FeatureStub()
        current = torch.tensor([[0, 0, 1]], dtype=torch.long)
        selected = ((2, 1, 3), (0, 0, 1))
        _, features, replay = Att_Diffuse_model.gereration_feature(
            None, SimpleNamespace(diffuc="add", delete_feature_reply=False),
            3, stub, None, None, None, None,
            np.array([[0, 0, 1]]), current, False, [],
            hcp_triples=selected)
        self.assertEqual({tuple(row) for row in replay.tolist()}, set(selected))
        self.assertTrue(torch.equal(stub.seen, replay))
        self.assertEqual(features.shape, (2, 3, 8))

    def test_probe_restores_model_and_rng_then_kd_backpropagates(self):
        torch.manual_seed(7)
        model = RecurrentRGCN(
            decoder_name="convtranse", encoder_name="uvrgcn", num_ents=4,
            num_rels=2, num_static_rels=0, num_words=0, h_dim=8,
            attn_heads=2, opn="sub", sequence_len=1, num_hidden_layers=1,
            use_static=False, entity_prediction=True, relation_prediction=True,
            use_cuda=False, gpu=-1, args=SimpleNamespace(mu=0.5, mu_r=0.5))
        teacher = copy.deepcopy(model)
        teacher.requires_grad_(False)
        graph = build_sub_graph(4, 2, np.array([[0, 0, 1], [1, 1, 2]]),
                                use_cuda=False, gpu=-1)
        triples = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        candidates = [HCPCandidate(0, ((0, 0, 1), (1, 1, 2))),
                      HCPCandidate(1, ((2, 2, 1),))]
        saved = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()

        risks = probe_hcp_risks(
            model, teacher, candidates, triples, [graph], None, False,
            task_weight=0.7, learning_rate=0.01, grad_norm=1.0,
            steps=2, score_batch_size=2)

        self.assertEqual(set(risks), {0, 1})
        self.assertTrue(all(np.isfinite(value) and value >= 0
                            for value in risks.values()))
        for key, value in saved.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

        replay = triples
        chunks = teacher_replay_logits(
            teacher, replay, [graph], None, False, batch_size=1)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(replay))
        model.train()
        loss = replay_kd_loss(
            model, chunks, replay, [graph], None, False,
            temperature=2.0, confidence=0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
