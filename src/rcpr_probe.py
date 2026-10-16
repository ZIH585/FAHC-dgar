"""Frozen, TRAIN-fitted low-rank probes; never update a backbone or read TEST."""
import math

import torch
from torch.nn import functional as F

from src.rcpr import ScoreResidual


def fit_head(query, base_scores, keys, targets, context, rank, steps, lr, mean_control=False, chunk=256):
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('probe steps and learning rate must be positive')
    # Common random initialization across controls, without advancing global RNG.
    head = ScoreResidual(query.shape[1], rank, chunk).to(query.device, query.dtype)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    query, base_scores, targets = query.detach(), base_scores.detach(), targets.detach()
    context = context.detached()
    first_loss = last_loss = None
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = head(base_scores, query, keys, context, mean_control)
            loss = F.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite probe loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            if step == 0:
                first_loss = float(loss.detach().cpu())
            last_loss = float(loss.detach().cpu())
    head.eval()
    return head, {'steps': steps, 'train_ce_initial': first_loss, 'train_ce_last_pre_step': last_loss,
                  'parameters': sum(p.numel() for p in head.parameters())}


def gate(stages, margin=.003, inverse_tolerance=.002):
    modes = ('mean-shared', 'mean-head', 'prototype', 'shuffled')
    valid = [s for s in stages if all(m in s.get('metrics', {}) for m in modes)]
    result = {'eligible_stages': len(valid), 'required_stages': len(stages),
              'margin_pp': margin * 100, 'inverse_tolerance_pp': inverse_tolerance * 100,
              'pass': False}
    if len(valid) != len(stages) or len(valid) < 3:
        result['reason'] = 'need at least three complete prespecified stages'
        return result
    average = {m: sum(s['metrics'][m]['mrr'] for s in valid) / len(valid) for m in modes}
    gaps = {m: average['prototype'] - average[m] for m in modes if m != 'prototype'}
    directional = {name: sum(s['metrics']['prototype'][name]['mrr'] -
        s['metrics']['mean-shared'][name]['mrr'] for s in valid) / len(valid)
        for name in ('forward', 'inverse')}
    improved = sum(s['metrics']['prototype']['mrr'] > s['metrics']['mean-shared']['mrr'] for s in valid)
    result.update(macro_stage_mrr=average, gain_vs_controls_pp={m:g*100 for m,g in gaps.items()},
                  direction_gain_vs_mean_shared_pp={k:v*100 for k,v in directional.items()},
                  improved_stages=improved, required_improved_stages=math.ceil(.6*len(valid)))
    result['pass'] = (all(g >= margin for g in gaps.values())
                      and improved >= result['required_improved_stages']
                      and directional['forward'] > 0 and directional['inverse'] >= -inverse_tolerance)
    return result
