import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN


class EntityDecoder(nn.Module):
    def forward(self, embedding, relation_embedding, triples):
        query = embedding[triples[:, 0]] + relation_embedding[triples[:, 1]]
        return query @ embedding.T


class TeacherKDTest(unittest.TestCase):
    def test_correct_confident_teacher_only_and_student_gradient(self):
        student = torch.tensor([[1.0, 0.0, -1.0],
                                [0.0, 1.0, -1.0],
                                [0.5, 0.0, -0.5]], requires_grad=True)
        teacher = torch.tensor([[4.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0],
                                [2.0, 0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 2, 0])
        temperature = 2.0

        loss = RecurrentRGCN.teacher_kl_loss(
            student, teacher, labels, temperature, confidence_threshold=0.9)
        expected = F.kl_div(
            F.log_softmax(student[:1] / temperature, dim=-1),
            F.softmax(teacher[:1].detach() / temperature, dim=-1),
            reduction="sum") * temperature ** 2 / 3
        self.assertTrue(torch.allclose(loss, expected))

        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad[0].abs().sum().item(), 0)
        self.assertEqual(student.grad[1:].abs().sum().item(), 0)

    def test_no_eligible_teacher_keeps_zero_student_gradient(self):
        student = torch.tensor([[0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[5.0, 0.0]])
        loss = RecurrentRGCN.teacher_kl_loss(
            student, teacher, torch.tensor([1]), confidence_threshold=0.5)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_entity_context_is_reused_without_reencoding(self):
        model = RecurrentRGCN.__new__(RecurrentRGCN)
        nn.Module.__init__(model)
        model.layer_norm = False
        model.num_ents = 3
        model.decoder_ob = EntityDecoder()
        entities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                requires_grad=True)
        relations = torch.tensor([[0.0, 0.0], [0.5, 0.0]])
        model.forward = Mock(return_value=([entities], None, relations, [], []))
        context = model.get_entity_context([object()], None, False)
        triples = torch.tensor([[0, 0, 1], [1, 1, 2]])

        first = model.get_entity_logits(None, triples[:1], None, False,
                                        context=context)
        second = model.get_entity_logits(None, triples[1:], None, False,
                                         context=context)
        combined = torch.cat([first, second])
        expected = model.decoder_ob(entities, relations, triples)
        self.assertTrue(torch.allclose(combined, expected))
        self.assertEqual(model.forward.call_count, 1)
        combined.sum().backward()
        self.assertGreater(entities.grad.abs().sum().item(), 0)

    def test_replay_loss_keeps_hcp_triples_on_cpu(self):
        model = RecurrentRGCN(
            decoder_name="convtranse", encoder_name="uvrgcn", num_ents=4,
            num_rels=2, num_static_rels=0, num_words=0, h_dim=8,
            attn_heads=2, opn="sub", sequence_len=1,
            num_hidden_layers=1, use_static=False,
            entity_prediction=True, relation_prediction=True,
            use_cuda=False, gpu=-1, args=SimpleNamespace(mu=0.5, mu_r=0.5))
        graph = build_sub_graph(4, 2, np.array([[0, 0, 1], [1, 1, 2]]),
                                use_cuda=False, gpu=-1)
        triples = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        context = model.get_entity_context([graph], None, False)
        logits = model.get_entity_logits(None, triples, None, False,
                                         context=context)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(logits.device.type, "cpu")

        losses = model.get_loss([graph], triples, None, False,
                                model_output=torch.randn(2, 3, 8),
                                model_output_x_t=torch.randn(2, 8),
                                tag=triples, diffuc="add",
                                output_reply_triple=triples)
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(torch.isfinite(x).all() for x in losses))
        self.assertEqual(model.get_all_triple(triples).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
