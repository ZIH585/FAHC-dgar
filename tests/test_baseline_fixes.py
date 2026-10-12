"""Small CPU checks for shared fixes, independent of the QCDR algorithm."""
import pickle
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from diffusion.model_21 import Att_Diffuse_model
from src.runtime import validate_history_cache


class BaselineFixTests(unittest.TestCase):
    def test_singleton_diffusion_ce(self):
        # Exercise the real loss method with its small required state.
        dummy = types.SimpleNamespace(loss_ce=torch.nn.CrossEntropyLoss(),
                                      emb_dim=3, temperature_object=.5)
        rep = torch.tensor([[.2, -.3, .7]], requires_grad=True)
        bank = torch.eye(3)
        bank = torch.cat((bank, torch.zeros(1, 3)))
        loss = Att_Diffuse_model.loss_diffu_ce(
            dummy, rep, torch.tensor([[0, 0, 1]]), None, emb_ent=bank)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(rep.grad.norm()), 0.)

    def test_cache_is_training_only_and_unchanged(self):
        train = [np.array([[0, 0, 1]], dtype=np.int64)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / 'snap_0.pkl'
            for history in ({'0': {(0, 0, 1)}},
                            {'0': {(0, 0, 1)}, '1': {(1, 2, 0)}}):
                data = pickle.dumps(history)
                path.write_bytes(data)
                validate_history_cache(train, 2, directory)
                self.assertEqual(path.read_bytes(), data)
            for history in ({'0': {(0, 0, 2)}}, {'1': {(0, 0, 1)}}, {}):
                path.write_bytes(pickle.dumps(history))
                with self.assertRaises(ValueError):
                    validate_history_cache(train, 2, directory)


if __name__ == '__main__':
    unittest.main()
