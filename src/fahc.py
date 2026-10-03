"""Probe-based forgetting scores and gated replay distillation for FAHC-DGAR."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from diffusion.fahc_selector import HCPCandidate, positive_loss_increase


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
                    steps: int = 1, score_batch_size: int = 256):
    """Probe current data, score positive HCP loss changes, then fully restore.

    A separate optimizer leaves the main Adam moments untouched. The saved
    state includes BatchNorm buffers, and RNG restoration makes the probe
    invisible to the subsequent DGAR training trajectory.
    """
    if steps < 1 or score_batch_size < 1:
        raise ValueError("probe steps and score batch size must be positive")
    if not 0 <= task_weight <= 1 or learning_rate <= 0 or grad_norm <= 0:
        raise ValueError("invalid probe optimization settings")
    if not any(candidate.triples for candidate in candidates):
        return {candidate.time_index: 0.0 for candidate in candidates}

    # Keep the temporary checkpoint on CPU instead of doubling GPU weights.
    state = {name: value.detach().cpu().clone()
             for name, value in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    old_was_training = old_model.training
    try:
        old_losses = _candidate_losses(old_model, candidates, history_glist,
                                       static_graph, use_cuda, score_batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=1e-5)
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
    finally:
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        old_model.train(old_was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return risks


@torch.no_grad()
def teacher_replay_logits(old_model, triples, history_glist, static_graph,
                          use_cuda: bool, batch_size: int = 256):
    """Cache old-model logits on CPU once per snapshot, in bounded chunks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    old_model.eval()
    context = old_model.get_entity_context(history_glist, static_graph, use_cuda)
    triples = torch.as_tensor(triples, dtype=torch.long)
    return [_scoring_logits(
        old_model, triples[start:start + batch_size], history_glist,
        static_graph, use_cuda, context).cpu()
        for start in range(0, len(triples), batch_size)]


def replay_kd_loss(model, teacher_chunks, triples, history_glist, static_graph,
                   use_cuda: bool, temperature: float, confidence: float):
    """Mean gated entity KL over replay facts with matched eval-mode logits.

    Eval mode keeps BatchNorm and dropout behavior consistent with the frozen
    teacher, while gradients still flow through the student encoder/decoder.
    """
    was_training = model.training
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
        model.train(was_training)
