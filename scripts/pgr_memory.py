"""Execution-only memory policies; model/checkpoint source fingerprints stay intact."""
from contextlib import contextmanager, ExitStack, nullcontext
import functools

import torch
from torch.utils.checkpoint import checkpoint


class RecomputeBuffers:
    """Give replayed forwards temporary BN buffers; preserve the live originals.

    Swapping references avoids in-place version changes to tensors saved by BN.
    Keep TRAIN mode: eval-mode BN would change both outputs and gradients.
    """
    def __init__(self, module):
        self.module, self.saved = module, []

    def __enter__(self):
        try:
            for module in self.module.modules():
                for name, value in list(module._buffers.items()):
                    if value is not None:
                        replacement = value.detach().clone()
                        self.saved.append((module, name, value))
                        module._buffers[name] = replacement
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for module, name, value in reversed(self.saved):
            module._buffers[name] = value
        self.saved.clear()


@contextmanager
def checkpoint_entity_decoder():
    """Recompute ConvTransE activations, preserving full-batch BN and dropout.

    Entity logits and CE remain the original FP32/full-candidate computations.
    Non-reentrant checkpoint also supports Original guidance's autograd.grad.
    """
    from src.decoder import ConvTransE
    original = ConvTransE.forward

    @functools.wraps(original)
    def forward(module, *args, **kwargs):
        if not module.training or not torch.is_grad_enabled():
            return original(module, *args, **kwargs)
        return checkpoint(lambda *a, **kw: original(module, *a, **kw), *args,
            use_reentrant=False, preserve_rng_state=True,
            context_fn=lambda: (nullcontext(), RecomputeBuffers(module)), **kwargs)

    ConvTransE.forward = forward
    try:
        yield
    finally:
        ConvTransE.forward = original


@contextmanager
def selective_offload(min_mib=64, pin_memory=False):
    """Offload only large CUDA saved tensors; keep small tensors on their device."""
    minimum = int(min_mib * 1024 * 1024)
    cpu = torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)

    def pack(tensor):
        if tensor.device.type == 'cuda' and tensor.numel() * tensor.element_size() >= minimum:
            return cpu.pack_hook(tensor)
        # Do not retain a grad_fn/reference cycle through the packed object.
        return tensor.detach()

    def unpack(value):
        return value if torch.is_tensor(value) else cpu.unpack_hook(value)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@contextmanager
def memory_policy(mode='cpu', min_mib=64, pin_memory=False):
    with ExitStack() as stack:
        if mode not in ('cpu', 'checkpoint', 'selective', 'checkpoint-selective', 'none'):
            raise ValueError(f'unknown memory policy: {mode}')
        if mode in ('checkpoint', 'checkpoint-selective'):
            stack.enter_context(checkpoint_entity_decoder())
        if mode == 'cpu':
            stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=pin_memory))
        elif mode in ('selective', 'checkpoint-selective'):
            stack.enter_context(selective_offload(min_mib, pin_memory))
        yield
