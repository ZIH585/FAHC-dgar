import random
import unittest
from unittest import mock

import numpy as np
import torch

from src.fahc_runtime import cpu_optimizer_state, isolated_rng


class IsolatedRNGTests(unittest.TestCase):
    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()

    def tearDown(self):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)

    def assert_numpy_state_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])

    def test_same_seed_repeats_and_restores_all_cpu_streams(self):
        random.seed(41)
        np.random.seed(41)
        torch.default_generator.manual_seed(41)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        draws = []
        for _ in range(2):
            with isolated_rng(2**32 + 13, torch.device("cpu")):
                draws.append((random.random(), float(np.random.random()),
                              float(torch.rand(()))))
            self.assertEqual(random.getstate(), before[0])
            self.assert_numpy_state_equal(np.random.get_state(), before[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(draws[0], draws[1])

    def test_exception_restores_states(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with isolated_rng(7, "cpu"):
                random.random()
                np.random.random()
                torch.rand(())
                raise RuntimeError("test failure")
        self.assertEqual(random.getstate(), before[0])
        self.assert_numpy_state_equal(np.random.get_state(), before[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_cpu_never_calls_cuda_rng_api(self):
        with (mock.patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError),
              mock.patch.object(torch.cuda, "set_rng_state", side_effect=AssertionError),
              mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError),
              mock.patch.object(torch.cuda, "manual_seed", side_effect=AssertionError),
              mock.patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError)):
            with isolated_rng(23, "cpu"):
                self.assertIsInstance(float(torch.rand(())), float)

    def test_cuda_request_touches_only_requested_generator(self):
        other = mock.Mock()
        requested = mock.Mock()
        saved_cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (mock.patch.object(torch.cuda, "default_generators", (other, requested)),
              mock.patch.object(torch.cuda, "get_rng_state",
                                return_value=saved_cuda_state) as get_state,
              mock.patch.object(torch.cuda, "set_rng_state") as set_state,
              mock.patch.object(torch.cuda, "manual_seed_all",
                                side_effect=AssertionError)):
            with isolated_rng(29, torch.device("cuda:1")):
                pass
        get_state.assert_called_once_with(torch.device("cuda:1"))
        set_state.assert_called_once_with(saved_cuda_state,
                                          torch.device("cuda:1"))
        requested.manual_seed.assert_called_once_with(29)
        other.manual_seed.assert_not_called()


class OptimizerStateTests(unittest.TestCase):
    def test_nested_state_is_detached_cloned_and_independent(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.tensor([3.0, 4.0], requires_grad=True),
            "nested": {"list": [torch.tensor(5.0)],
                       "tuple": (torch.tensor(6.0),)},
        }
        saved = cpu_optimizer_state(optimizer)
        state = next(iter(saved["state"].values()))
        self.assertEqual(saved["param_groups"][0]["lr"], 0.01)
        self.assertEqual(state["exp_avg"].device.type, "cpu")
        self.assertFalse(state["exp_avg"].requires_grad)

        optimizer.state[parameter]["exp_avg"].data[0] = 99.0
        optimizer.state[parameter]["nested"]["list"][0].fill_(99.0)
        self.assertEqual(float(state["exp_avg"][0]), 3.0)
        self.assertEqual(float(state["nested"]["list"][0]), 5.0)

        state["nested"]["tuple"][0].fill_(88.0)
        saved["param_groups"][0]["lr"] = 0.5
        self.assertEqual(float(optimizer.state[parameter]["nested"]["tuple"][0]),
                         6.0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_saved_state_loads_without_optimizer_steps(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.tensor([3.0, 4.0]),
            "exp_avg_sq": torch.tensor([5.0, 6.0]),
        }
        saved = cpu_optimizer_state(optimizer)
        replacement = torch.nn.Parameter(torch.tensor([7.0, 8.0]))
        restored = torch.optim.Adam([replacement], lr=0.5)
        restored.load_state_dict(saved)
        self.assertEqual(restored.param_groups[0]["lr"], 0.01)
        self.assertTrue(torch.equal(restored.state[replacement]["exp_avg"],
                                    torch.tensor([3.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
