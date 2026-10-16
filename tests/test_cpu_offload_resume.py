"""CPU checks for exact resume settings and saved-tensor offload semantics."""
import contextlib
import copy
import io
import unittest

import torch

from scripts.resume_pgr_cpu_offload import build_resume_args
from src.runtime import rng_state, restore_rng, settings
import test_pgr


class CPUOffloadResumeTests(unittest.TestCase):
    def test_saved_settings_are_preserved(self):
        fixture = test_pgr.PGRTests()
        saved = settings(fixture.args("mean-shared"))
        args = build_resume_args({"settings": saved}, "/tmp/pgr-offload", 1, "/tmp/data")
        self.assertEqual(settings(args), saved)
        self.assertEqual(args.resume, args.run_dir / "latest.pt")
        self.assertTrue(args.final_validation)
        self.assertTrue(args.pgr_diagnostics)
        invalid = dict(saved, lr=-1)
        with self.assertRaises(ValueError):
            build_resume_args({"settings": invalid}, "/tmp/pgr-offload", 1, "/tmp/data")

    def test_offload_preserves_live_predictor_loss_gradients_buffers_and_rng(self):
        fixture = test_pgr.PGRTests()
        model = fixture.model(fixture.args("mean-shared"))
        models = [model, copy.deepcopy(model)]
        tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
        features = torch.randn(3, 3, 8)
        state = rng_state("cpu")
        results = []
        for index, current in enumerate(models):
            restore_rng(state, "cpu")
            context = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                       if index else contextlib.nullcontext())
            with context, contextlib.redirect_stdout(io.StringIO()):
                losses = current.get_loss(fixture.history(), tags[:1], None, False,
                    model_output=features, model_output_x_t=features[:, 2], tag=tags,
                    diffuc="add", output_reply_triple=tags)
                loss = .7 * losses[0] + .3 * losses[1] + losses[2]
                loss.backward()
            results.append((loss.detach(), torch.get_rng_state().clone()))
        for actual, expected in zip(results[0], results[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fixture.assert_state(models[0].state_dict(), models[1].state_dict())
        for (name, left), (_, right) in zip(models[0].named_parameters(), models[1].named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
