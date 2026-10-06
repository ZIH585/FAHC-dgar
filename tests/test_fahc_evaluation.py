"""Exercise final evaluation with empty HCPs; no training or CUDA calls."""

import io
import os
import pickle
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.main import test as evaluate


class EvaluationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.observations = []

    def predict(self, history, num_rels, static, triples, features,
                objects, replay, use_cuda, history_graph):
        self.observations.append((
            len(history), history_graph, tuple(replay.shape), replay.dtype,
            random.random(), float(np.random.random()), float(torch.rand(()))))
        inverse = triples[:, [2, 1, 0]].clone()
        inverse[:, 1] += num_rels
        queries = torch.cat((triples, inverse))
        return queries, torch.zeros(len(queries), 4), torch.zeros(len(queries), 2)


class FAHCEvaluationTest(unittest.TestCase):
    def test_empty_hcp_and_long_test_history_restore_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            cache = root / "data" / "Tiny" / "history_snap_v2"
            cache.mkdir(parents=True)
            for index in range(4):
                with (cache / f"snap_{index}.pkl").open("wb") as stream:
                    pickle.dump({}, stream)
            previous_cwd = os.getcwd()
            try:
                os.chdir(root / "src")
                args = SimpleNamespace(
                    seed=13, gpu=-1, dataset="Tiny", reply_batch=2,
                    train_history_len=3, test_history_len=5,
                    diffuc="add", delete_feature_reply=False)
                model = EvaluationModel()
                diffusion = Mock(side_effect=AssertionError("empty HCP invoked diffusion"))
                snapshots = [np.array([[0, 0, 1]]) for _ in range(5)]
                python_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state().clone()

                def rank(*args, **kwargs):
                    ranks = torch.ones(2, dtype=torch.long)
                    return torch.tensor(1.0), torch.tensor(1.0), ranks, ranks

                with (patch("src.main.build_sub_graph", return_value=object()),
                      patch("src.main.utils.get_total_rank", side_effect=rank),
                      patch.object(torch.cuda, "get_rng_state",
                                   side_effect=AssertionError("CPU evaluation touched CUDA")),
                      patch.object(torch.cuda, "set_rng_state",
                                   side_effect=AssertionError("CPU evaluation touched CUDA")),
                      redirect_stdout(io.StringIO())):
                    for _ in range(2):
                        result = evaluate(
                            args, model, snapshots, snapshots, 1, 4, False,
                            [{}] * 5, [{}] * 5, "unused", None, "valid",
                            None, None, None, diffusion)

                self.assertEqual(len(result), 14)
                self.assertEqual(len(result[6]), 4)
                self.assertEqual([row[0] for row in model.observations[:4]],
                                 [1, 2, 3, 4])
                self.assertTrue(all(row[1] is None for row in model.observations))
                self.assertTrue(all(row[2] == (0, 3) and row[3] == torch.long
                                    for row in model.observations))
                self.assertEqual(model.observations[:4], model.observations[4:])
                diffusion.assert_not_called()
                self.assertEqual(random.getstate(), python_state)
                restored_numpy = np.random.get_state()
                self.assertEqual(numpy_state[0], restored_numpy[0])
                self.assertTrue(np.array_equal(numpy_state[1], restored_numpy[1]))
                self.assertEqual(numpy_state[2:], restored_numpy[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
