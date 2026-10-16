"""CPU checks for exact resume settings and saved-tensor offload semantics."""
import contextlib
import copy
import io
import unittest

import torch

from scripts.resume_pgr_cpu_offload import build_resume_args
from scripts.pgr_memory import memory_policy, RecomputeBuffers
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
        modes = ('none', 'cpu', 'checkpoint', 'selective', 'checkpoint-selective')
        models = [model, *[copy.deepcopy(model) for _ in modes[1:]]]
        tags = torch.tensor([[0, 0, 1], [1, 2, 0], [2, 1, 3]])
        features = torch.randn(3, 3, 8)
        state = rng_state("cpu")
        results = []
        calls = []
        for mode, current in zip(modes, models):
            restore_rng(state, "cpu")
            count = [0]
            handle = current.decoder_ob.conv1.register_forward_hook(lambda *a: count.__setitem__(0, count[0] + 1))
            with memory_policy(mode), contextlib.redirect_stdout(io.StringIO()):
                losses = current.get_loss(fixture.history(), tags[:1], None, False,
                    model_output=features, model_output_x_t=features[:, 2], tag=tags,
                    diffuc="add", output_reply_triple=tags)
                loss = .7 * losses[0] + .3 * losses[1] + losses[2]
                loss.backward()
            handle.remove()
            calls.append(count[0])
            results.append((loss.detach(), torch.get_rng_state().clone()))
        self.assertGreater(calls[2], calls[0])
        self.assertGreater(calls[4], calls[0])
        for index in range(1, len(modes)):
            for actual, expected in zip(results[0], results[index]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            fixture.assert_state(models[0].state_dict(), models[index].state_dict())
            for (name, left), (_, right) in zip(models[0].named_parameters(), models[index].named_parameters()):
                if left.grad is None:
                    self.assertIsNone(right.grad, name)
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)

    def test_recompute_restores_buffer_references_on_exception(self):
        module = torch.nn.BatchNorm1d(2)
        before = {name: value for name, value in module._buffers.items()}
        state = copy.deepcopy(module.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            with RecomputeBuffers(module):
                module(torch.randn(3, 2))
                raise RuntimeError('expected')
        self.assertTrue(all(module._buffers[name] is value for name, value in before.items()))
        for name, value in state.items():
            torch.testing.assert_close(module.state_dict()[name], value, rtol=0, atol=0)

    def test_checkpoint_preserves_actual_guidance_autograd_grad_and_eval(self):
        import test_real_cpu_pipeline
        from src.decoder import ConvTransE
        original = ConvTransE.forward
        with memory_policy('checkpoint'):
            test_real_cpu_pipeline.RealCpuTests().test_frozen_no_grad_generators_match_upstream_train_mode()
        self.assertIs(ConvTransE.forward, original)


if __name__ == "__main__":
    unittest.main()
