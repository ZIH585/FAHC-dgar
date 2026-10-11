"""Probe-based forgetting scores and gated replay distillation for FAHC-DGAR."""

from __future__ import annotations

import copy
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from diffusion.fahc_selector import HCPCandidate, positive_loss_increase


_MISSING = object()
_TRANSIENT_KEYS = ("h", "h_0", "rel_emb")


def _batchnorm_state(model):
    """Copy only BatchNorm running statistics for diagnostic comparisons."""
    result = {}
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name, None)
                if value is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    result[key] = value.detach().cpu().clone()
    return result


@torch.no_grad()
def _load_batchnorm_state(model, state):
    buffers = dict(model.named_buffers())
    for name, value in state.items():
        buffers[name].copy_(value)


def _capture_runtime_state(model):
    """Save module modes and forward caches omitted from ``state_dict``."""
    return [(module, module.training,
             {key: module.__dict__.get(key, _MISSING)
              for key in _TRANSIENT_KEYS})
            for module in model.modules()]


def _restore_runtime_state(snapshot):
    # Assign each flag directly: Module.train() would overwrite mixed child modes.
    for module, training, caches in snapshot:
        module.training = training
        for key, value in caches.items():
            if value is _MISSING:
                module.__dict__.pop(key, None)
            else:
                module.__dict__[key] = value


def _probe_optimizer(model, learning_rate, training_optimizer):
    """Build an independent Adam with the current trainer's moments and groups."""
    if training_optimizer is None:
        return torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    if not isinstance(training_optimizer, torch.optim.Adam):
        raise TypeError("training_optimizer must be a torch.optim.Adam")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    groups = []
    for group in training_optimizer.param_groups:
        parameters = list(group["params"])
        if any(id(parameter) not in model_parameters for parameter in parameters):
            raise ValueError("training_optimizer contains parameters from another model")
        groups.append({"params": parameters})
    optimizer = torch.optim.Adam(groups, lr=learning_rate)
    optimizer.load_state_dict(copy.deepcopy(training_optimizer.state_dict()))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return optimizer


def _scoring_logits(model, triples, history_glist, static_graph, use_cuda,
                    context):
    """Use the regular eval decoder path even for a one-fact chunk.

    ConvTransE skips its second BatchNorm for a singleton, including in eval
    mode. Duplicating only the decoder input preserves its normal eval path;
    the duplicate output is discarded and contributes no loss or gradient.
    """
    if len(triples) == 1:
        doubled = torch.cat((triples, triples), dim=0)
        return model.get_entity_logits(
            history_glist, doubled, static_graph, use_cuda,
            context=context)[:1]
    return model.get_entity_logits(
        history_glist, triples, static_graph, use_cuda, context=context)


@torch.no_grad()
def _candidate_losses(model, candidates: Sequence[HCPCandidate], history_glist,
                      static_graph, use_cuda: bool, batch_size: int):
    """Score HCP facts under the same current history used for DGAR replay."""
    model.eval()
    context = model.get_entity_context(history_glist, static_graph, use_cuda)
    device = context[0].device
    losses = {}
    for candidate in candidates:
        parts = []
        for start in range(0, len(candidate.triples), batch_size):
            triples = torch.as_tensor(candidate.triples[start:start + batch_size],
                                      dtype=torch.long, device=device)
            logits = _scoring_logits(model, triples, history_glist,
                                    static_graph, use_cuda, context)
            parts.extend(F.cross_entropy(logits, triples[:, 2],
                                         reduction="none").cpu().tolist())
        losses[candidate.time_index] = parts
    return losses


def probe_hcp_risks(model, old_model, candidates: Sequence[HCPCandidate],
                    current_triples, history_glist, static_graph, use_cuda: bool,
                    task_weight: float, learning_rate: float, grad_norm: float,
                    steps: int = 1, score_batch_size: int = 256,
                    training_optimizer=None, diagnostics=None):
    """Probe current data, score positive HCP loss changes, then fully restore.

    A separate Adam copies the main optimizer's moments and group settings
    when supplied, changing only its learning rate. The saved model state
    includes BatchNorm buffers; RNG restoration leaves the subsequent DGAR
    training trajectory unchanged.
    """
    if steps < 1 or score_batch_size < 1:
        raise ValueError("probe steps and score batch size must be positive")
    if not 0 <= task_weight <= 1 or learning_rate <= 0 or grad_norm <= 0:
        raise ValueError("invalid probe optimization settings")
    if not any(candidate.triples for candidate in candidates):
        if diagnostics is not None:
            diagnostics.update({
                "old_losses": {candidate.time_index: [] for candidate in candidates},
                "after_losses": {candidate.time_index: [] for candidate in candidates},
                "signed_increase": {candidate.time_index: 0.0 for candidate in candidates},
                "fixed_bn_risks": {candidate.time_index: 0.0 for candidate in candidates},
            })
        return {candidate.time_index: 0.0 for candidate in candidates}

    # Keep the temporary checkpoint on CPU instead of doubling GPU weights.
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    old_bn = _batchnorm_state(model) if diagnostics is not None else None
    cpu_rng = torch.get_rng_state()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    model_device = next(model.parameters()).device
    cuda_rng = (torch.cuda.get_rng_state(model_device)
                if model_device.type == "cuda" else None)
    model_runtime = _capture_runtime_state(model)
    old_runtime = _capture_runtime_state(old_model)
    gradients = [(parameter, parameter.grad)
                 for parameter in model.parameters()]
    try:
        old_losses = _candidate_losses(old_model, candidates, history_glist,
                                       static_graph, use_cuda, score_batch_size)
        optimizer = _probe_optimizer(model, learning_rate, training_optimizer)
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_e, loss_r, loss_static, _, _ = model.get_loss(
                history_glist, current_triples, static_graph, use_cuda)
            loss = task_weight * loss_e + (1 - task_weight) * loss_r + loss_static
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
        probe_losses = _candidate_losses(model, candidates, history_glist,
                                         static_graph, use_cuda, score_batch_size)
        risks = {candidate.time_index: positive_loss_increase(
            old_losses[candidate.time_index], probe_losses[candidate.time_index])
            for candidate in candidates}
        if diagnostics is not None:
            diagnostics.update({
                "old_losses": old_losses,
                "after_losses": probe_losses,
                "signed_increase": {
                    index: (sum(new - old for old, new in zip(values, probe_losses[index]))
                            / len(values) if values else 0.0)
                    for index, values in old_losses.items()},
            })
            # Keep the probe weights, replacing only its BN statistics.
            _load_batchnorm_state(model, old_bn)
            fixed_bn_losses = _candidate_losses(
                model, candidates, history_glist, static_graph, use_cuda,
                score_batch_size)
            diagnostics["fixed_bn_risks"] = {
                candidate.time_index: positive_loss_increase(
                    old_losses[candidate.time_index],
                    fixed_bn_losses[candidate.time_index])
                for candidate in candidates}
    finally:
        model.load_state_dict(state)
        for parameter, gradient in gradients:
            parameter.grad = gradient
        _restore_runtime_state(model_runtime)
        _restore_runtime_state(old_runtime)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, model_device)
    return risks


@torch.no_grad()
def teacher_replay_logits(old_model, triples, history_glist, static_graph,
                          use_cuda: bool, batch_size: int = 256):
    """Cache old-model logits on CPU once per snapshot, in bounded chunks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    runtime = _capture_runtime_state(old_model)
    try:
        old_model.eval()
        context = old_model.get_entity_context(history_glist, static_graph, use_cuda)
        triples = torch.as_tensor(triples, dtype=torch.long)
        return [_scoring_logits(
            old_model, triples[start:start + batch_size], history_glist,
            static_graph, use_cuda, context).cpu()
            for start in range(0, len(triples), batch_size)]
    finally:
        _restore_runtime_state(runtime)


def replay_kd_loss(model, teacher_chunks, triples, history_glist, static_graph,
                   use_cuda: bool, temperature: float, confidence: float):
    """Mean gated entity KL over replay facts with matched eval-mode logits.

    Eval mode keeps BatchNorm and dropout behavior consistent with the frozen
    teacher, while gradients still flow through the student encoder/decoder.
    """
    runtime = _capture_runtime_state(model)
    model.eval()
    try:
        context = model.get_entity_context(history_glist, static_graph, use_cuda)
        device = context[0].device
        triples = torch.as_tensor(triples, dtype=torch.long, device=device)
        if len(triples) == 0:
            return context[0].sum() * 0
        total = context[0].sum() * 0
        offset = 0
        for teacher in teacher_chunks:
            batch = triples[offset:offset + len(teacher)]
            student = _scoring_logits(model, batch, history_glist,
                                      static_graph, use_cuda, context)
            # teacher_kl_loss averages over the batch, including gated-out facts.
            total = total + len(batch) * model.teacher_kl_loss(
                student, teacher, batch[:, 2], temperature, confidence)
            offset += len(batch)
        if offset != len(triples):
            raise ValueError("teacher logits and replay facts have different lengths")
        return total / len(triples)
    finally:
        _restore_runtime_state(runtime)
