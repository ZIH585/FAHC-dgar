"""CPU graph and state checks; every optimizer step is mocked or omitted."""

import copy
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from diffusion.fahc_selector import HCPCandidate
from diffusion.difffu_21 import DiffuRec
from diffusion.model_21 import Att_Diffuse_model
from diffusion.regcn import RGCNCell as DiffusionRGCNCell
from rgcn.utils import build_sub_graph
from src.fahc import probe_hcp_risks, replay_kd_loss
from src.rrgcn import RecurrentRGCN


def tiny_model():
    return RecurrentRGCN(
        decoder_name="convtranse", encoder_name="uvrgcn", num_ents=4,
        num_rels=2, num_static_rels=1, num_words=1, h_dim=8,
        attn_heads=2, opn="sub", sequence_len=3, num_bases=2,
        num_hidden_layers=1, dropout=0.2, self_loop=True,
        input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2,
        use_static=True, discount=1, angle=10,
        entity_prediction=True, relation_prediction=True,
        use_cuda=False, gpu=-1, args=SimpleNamespace(mu=0.5, mu_r=0.5))


def tiny_graphs():
    history = build_sub_graph(
        4, 2, np.array([[0, 0, 1], [1, 1, 2]]), False, -1)
    static = build_sub_graph(
        5, 1, np.array([[0, 0, 4], [2, 0, 4]]), False, -1)
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
    return history, static, triples


class FAHCStateTest(unittest.TestCase):
    def test_probe_copies_training_adam_moments_and_group_settings(self):
        model = tiny_model()
        teacher = copy.deepcopy(model)
        history, static, triples = tiny_graphs()
        parameters = list(model.parameters())
        training_optimizer = torch.optim.Adam([
            {"params": parameters[:1], "lr": 0.004,
             "betas": (0.8, 0.9), "eps": 1e-7, "weight_decay": 0.02},
            {"params": parameters[1:], "lr": 0.006,
             "betas": (0.7, 0.95), "eps": 1e-6, "weight_decay": 0.03},
        ])
        first = parameters[0]
        training_optimizer.state[first] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(first, 0.25),
            "exp_avg_sq": torch.full_like(first, 0.125),
        }
        before = copy.deepcopy(training_optimizer.state_dict())

        with patch.object(torch.optim.Adam, "step", autospec=True,
                          return_value=None) as step:
            probe_hcp_risks(
                model, teacher, [HCPCandidate(0, ((0, 0, 1),))],
                triples, [history], static, False,
                task_weight=0.7, learning_rate=0.01, grad_norm=1.0,
                steps=1, score_batch_size=1,
                training_optimizer=training_optimizer)
        step.assert_called_once()
        probe_optimizer = step.call_args.args[0]
        self.assertIsNot(probe_optimizer, training_optimizer)
        self.assertEqual(len(probe_optimizer.param_groups), 2)
        for actual, original in zip(probe_optimizer.param_groups,
                                    training_optimizer.param_groups):
            self.assertEqual(actual["lr"], 0.01)
            for key in ("betas", "eps", "weight_decay", "amsgrad"):
                self.assertEqual(actual[key], original[key])
        probe_moment = probe_optimizer.state[first]["exp_avg"]
        original_moment = training_optimizer.state[first]["exp_avg"]
        self.assertTrue(torch.equal(probe_moment, original_moment))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertNotEqual(probe_optimizer.state[first][key].data_ptr(),
                                training_optimizer.state[first][key].data_ptr())
        probe_moment.add_(1)
        self.assertTrue(torch.equal(original_moment,
                                    before["state"][0]["exp_avg"]))
        self.assertEqual(training_optimizer.state_dict()["param_groups"],
                         before["param_groups"])
        self.assertTrue(torch.equal(training_optimizer.state[first]["step"],
                                    before["state"][0]["step"]))
        self.assertTrue(torch.equal(training_optimizer.state[first]["exp_avg_sq"],
                                    before["state"][0]["exp_avg_sq"]))

    def test_probe_restores_all_state_without_real_optimizer_step(self):
        model = tiny_model()
        teacher = copy.deepcopy(model)
        history, static, triples = tiny_graphs()
        model.train()
        teacher.train()
        model.decoder_ob.bn0.eval()
        teacher.decoder_ob.bn1.eval()
        h_before = torch.tensor([9.0])
        h0_before = torch.tensor([8.0])
        rel_before = torch.tensor([7.0])
        teacher_h_before = torch.tensor([6.0])
        model.h = h_before
        model.h_0 = h0_before
        model.rgcn.layers[0].rel_emb = rel_before
        teacher.h = teacher_h_before
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        original_grad = parameter.grad
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        modes = {key: module.training for key, module in model.named_modules()}
        teacher_modes = {key: module.training
                         for key, module in teacher.named_modules()}
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()

        original_get_loss = model.get_loss

        def consuming_get_loss(*args, **kwargs):
            random.random()
            np.random.random()
            return original_get_loss(*args, **kwargs)

        model.get_loss = consuming_get_loss
        with (patch.object(torch.optim.Adam, "step", return_value=None) as step,
              patch.object(torch.cuda, "get_rng_state",
                           side_effect=AssertionError("CPU probe touched CUDA")),
              patch.object(torch.cuda, "set_rng_state",
                           side_effect=AssertionError("CPU probe touched CUDA"))):
            risks = probe_hcp_risks(
                model, teacher,
                [HCPCandidate(0, ((0, 0, 1), (1, 1, 2)))],
                triples, [history], static, False,
                task_weight=0.7, learning_rate=0.01, grad_norm=1.0,
                steps=1, score_batch_size=1)
        step.assert_called_once()
        self.assertTrue(math.isfinite(risks[0]) and risks[0] >= 0)
        self.assertTrue(all(torch.equal(weights[key], value)
                            for key, value in model.state_dict().items()))
        self.assertEqual(modes, {key: module.training
                                 for key, module in model.named_modules()})
        self.assertEqual(teacher_modes, {key: module.training
                                         for key, module in teacher.named_modules()})
        self.assertIs(model.h, h_before)
        self.assertIs(model.h_0, h0_before)
        self.assertIs(model.rgcn.layers[0].rel_emb, rel_before)
        self.assertIs(teacher.h, teacher_h_before)
        self.assertIs(parameter.grad, original_grad)
        self.assertEqual(python_rng, random.getstate())
        restored_numpy_rng = np.random.get_state()
        self.assertEqual(numpy_rng[0], restored_numpy_rng[0])
        self.assertTrue(np.array_equal(numpy_rng[1], restored_numpy_rng[1]))
        self.assertEqual(numpy_rng[2:], restored_numpy_rng[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_probe_exception_restores_mutated_state_without_optimizer_step(self):
        model = tiny_model()
        teacher = copy.deepcopy(model)
        history, static, triples = tiny_graphs()
        model.train()
        teacher.train()
        model.decoder_ob.bn0.eval()
        teacher.decoder_ob.bn1.eval()
        first = next(model.parameters())
        first.grad = torch.ones_like(first)
        original_grad = first.grad
        model.h = torch.tensor([1.0])
        model.h_0 = torch.tensor([2.0])
        model.rgcn.layers[0].rel_emb = torch.tensor([3.0])
        teacher.h = torch.tensor([4.0])
        caches = (model.h, model.h_0, model.rgcn.layers[0].rel_emb,
                  teacher.h)
        model_state = {key: value.clone() for key, value in model.state_dict().items()}
        teacher_state = {key: value.clone()
                         for key, value in teacher.state_dict().items()}
        model_modes = {key: module.training for key, module in model.named_modules()}
        teacher_modes = {key: module.training
                         for key, module in teacher.named_modules()}
        main_optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
        main_optimizer.state[first] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(first, 0.25),
            "exp_avg_sq": torch.full_like(first, 0.125),
        }
        optimizer_state = copy.deepcopy(main_optimizer.state_dict())
        moment_ref = main_optimizer.state[first]["exp_avg"]
        random.seed(79)
        np.random.seed(79)
        torch.manual_seed(79)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        score_calls = []

        def scoring(which_model, *args, **kwargs):
            score_calls.append(which_model)
            if which_model is teacher:
                teacher.eval()
                teacher.h = torch.tensor([40.0])
                return {0: [0.5]}
            with torch.no_grad():
                first.add_(5)
                model.decoder_ob.bn0.running_mean.add_(7)
                model.decoder_ob.bn0.num_batches_tracked.add_(5)
            model.eval()
            model.decoder_ob.bn1.train()
            model.h = torch.tensor([10.0])
            model.h_0 = torch.tensor([20.0])
            model.rgcn.layers[0].rel_emb = torch.tensor([30.0])
            first.grad = torch.full_like(first, 42.0)
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("synthetic scoring failure")

        with (patch("src.fahc._candidate_losses", side_effect=scoring),
              patch.object(torch.optim.Adam, "step", return_value=None) as step):
            with self.assertRaisesRegex(RuntimeError, "synthetic scoring failure"):
                probe_hcp_risks(
                    model, teacher, [HCPCandidate(0, ((0, 0, 1),))],
                    triples, [history], static, False,
                    task_weight=0.7, learning_rate=0.01, grad_norm=1.0,
                    steps=1, score_batch_size=1,
                    training_optimizer=main_optimizer)
        step.assert_called_once()
        self.assertEqual(score_calls, [teacher, model])
        self.assertTrue(all(torch.equal(model_state[key], value)
                            for key, value in model.state_dict().items()))
        self.assertTrue(all(torch.equal(teacher_state[key], value)
                            for key, value in teacher.state_dict().items()))
        self.assertEqual(model_modes, {key: module.training
                                       for key, module in model.named_modules()})
        self.assertEqual(teacher_modes, {key: module.training
                                         for key, module in teacher.named_modules()})
        self.assertIs(first.grad, original_grad)
        self.assertIs(model.h, caches[0])
        self.assertIs(model.h_0, caches[1])
        self.assertIs(model.rgcn.layers[0].rel_emb, caches[2])
        self.assertIs(teacher.h, caches[3])
        self.assertEqual(python_rng, random.getstate())
        restored_numpy_rng = np.random.get_state()
        self.assertEqual(numpy_rng[0], restored_numpy_rng[0])
        self.assertTrue(np.array_equal(numpy_rng[1], restored_numpy_rng[1]))
        self.assertEqual(numpy_rng[2:], restored_numpy_rng[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertIs(main_optimizer.state[first]["exp_avg"], moment_ref)
        self.assertEqual(main_optimizer.state_dict()["param_groups"],
                         optimizer_state["param_groups"])
        for key, value in optimizer_state["state"][0].items():
            self.assertTrue(torch.equal(main_optimizer.state[first][key], value))

    def test_static_self_loop_ce_and_nonzero_kd_backward(self):
        model = tiny_model()
        history, static, triples = tiny_graphs()
        histories = [history,
                     build_sub_graph(4, 2, np.array([[1, 0, 2], [2, 1, 3]]),
                                     False, -1),
                     build_sub_graph(4, 2, np.array([[0, 1, 3], [3, 0, 2]]),
                                     False, -1)]
        features = torch.randn(2, 3, 8)
        model.train()
        entity, relation, constraint, _, _ = model.get_loss(
            histories, triples, static, False,
            model_output=features, model_output_x_t=features[:, -1, :],
            tag=triples, diffuc="add", output_reply_triple=triples)
        base_loss = 0.7 * entity + 0.3 * relation + constraint

        # Mix child modes to check that the KD eval pass restores each one.
        model.decoder_ob.bn0.eval()
        modes = {key: module.training for key, module in model.named_modules()}
        bn_before = {key: value.clone() for key, value in model.state_dict().items()
                     if "running_" in key or "num_batches_tracked" in key}
        teacher_logits = torch.full((2, 4), -4.0)
        teacher_logits[0, 1] = 8.0
        teacher_logits[1, 2] = 8.0
        kd = replay_kd_loss(
            model, [teacher_logits[:1], teacher_logits[1:]], triples,
            histories, static, False, temperature=2.0, confidence=0.0)
        self.assertGreater(kd.item(), 0)
        self.assertEqual(modes, {key: module.training
                                 for key, module in model.named_modules()})
        self.assertTrue(all(torch.equal(bn_before[key], value)
                            for key, value in model.state_dict().items()
                            if key in bn_before))

        (base_loss + kd).backward()
        self.assertGreater(model.decoder_ob.conv1.weight.grad.norm().item(), 0)
        self.assertIsNotNone(model.rgcn.layers[0].weight_neighbor.grad)
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_diffusion_self_loop_graph_backward_on_cpu(self):
        cell = DiffusionRGCNCell(
            num_nodes=4, h_dim=8, out_dim=8, num_rels=4,
            num_hidden_layers=1, self_loop=True,
            encoder_name="uvrgcn", opn="sub", use_cuda=False)
        history, _, _ = tiny_graphs()
        entity = torch.randn(4, 8, requires_grad=True)
        relation = torch.randn(4, 8, requires_grad=True)
        output = cell(history, entity, [relation, relation])
        output.square().sum().backward()
        self.assertGreater(entity.grad.norm().item(), 0)
        self.assertGreater(relation.grad.norm().item(), 0)

    def test_static_diffusion_forward_backward_without_step(self):
        args = SimpleNamespace(
            hidden_size=8, heads=2, dropout=0.2, num_blocks=1,
            num_blocks_cross=0, lambda_uncertainty=0.01,
            pattern_noise_radio=1.0, k_step=0, num_rels=2, num_nodes=4,
            schedule_sampler_name="uniform", diffusion_steps=10,
            noise_schedule="linear", rescale_timesteps=False,
            scale=1.0, beta_start=0.0001, beta_end=0.02,
            max_len=8, add_static_graph=True,
            temperature_object=0.5, gpu=-1)
        model = Att_Diffuse_model(
            DiffuRec(args), args, "uvrgcn", 4, 2,
            num_bases=2, dropout=0.2, max_time=4,
            num_words=1, num_static_rels=1)
        _, static, replay = tiny_graphs()
        queries = torch.tensor([[0, 0, 1], [1, 1, 2],
                                [1, 2, 0], [2, 3, 1]], dtype=torch.long)
        result = model(
            None, queries, args, True, False, static_graph=static, ct=2,
            model_output=torch.randn(2, 3, 8), targets=replay)
        loss = model.loss_diffu_ce(
            result[1], queries, result[7], queries, result[4], result[5])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.statci_rgcn_layer.weight.grad)
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                            if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()
