"""Balance DGAR's two entity replay losses using their shared-representation gradients.

No second forward, parameter-gradient mutation, extra teacher or persistent memory.
The detached EMA resets at every chronological stage, including after resume.
"""
from dataclasses import dataclass
import json
import math

import torch


@dataclass(frozen=True)
class BalanceConfig:
    mode: str = 'original'
    fixed_weight: float = 0.5
    min_weight: float = 0.5
    target_ratio: float = 1.0
    conflict_scale: float = 1.0
    ema: float = 0.8

    def __post_init__(self):
        if self.mode not in ('original', 'fixed', 'adaptive'):
            raise ValueError('grb-mode must be original, fixed or adaptive')
        if not all(math.isfinite(x) for x in (self.fixed_weight, self.min_weight,
                self.target_ratio, self.conflict_scale, self.ema)):
            raise ValueError('GRB parameters must be finite')
        if not 0 <= self.fixed_weight <= 1 or not 0 <= self.min_weight <= 1:
            raise ValueError('GRB weights must be in [0,1]')
        if self.target_ratio <= 0 or self.conflict_scale < 0 or not 0 <= self.ema < 1:
            raise ValueError('GRB requires ratio > 0, conflict >= 0, EMA in [0,1)')

    @classmethod
    def from_args(cls, args):
        return cls(args.grb_mode, args.grb_fixed_weight, args.grb_min_weight,
                   args.grb_target_ratio, args.grb_conflict_scale, args.grb_ema)


def weight_from_moments(moments, config):
    """moments = [||g_current||^2, ||g_replay||^2, dot(g_current,g_replay)].

    Inputs represent entity CE gradients w.r.t. the final N x d entity state,
    before task weighting, clipping and Adam. They are not parameter gradients.
    Invalid/zero information conservatively falls back to Original weight 1.
    """
    values = moments.detach().float()
    a, b, dot = values.unbind()
    nc, nr = a.clamp_min(0).sqrt(), b.clamp_min(0).sqrt()
    eps = torch.finfo(values.dtype).tiny
    cosine = (dot / (nc * nr).clamp_min(eps)).clamp(-1, 1)
    raw = config.target_ratio * nc / nr.clamp_min(eps)
    raw = raw / (1 + config.conflict_scale * (-cosine).clamp_min(0))
    weight = raw.clamp(config.min_weight, 1)
    valid = torch.isfinite(values).all() & (nc > eps) & (nr > eps)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return torch.where(valid, weight, torch.ones_like(weight)), cosine, valid


class ReplayBalancer:
    """A training-only controller, intentionally outside nn.Module/state_dict.

    Latest checkpoints are saved only after a whole stage. No control state
    crosses a stage boundary, so the existing model/Adam/RNG resume is sufficient.
    Stage records contain detached scalar tensors and are flushed once per stage.
    """
    fields = ('weight', 'current_loss', 'replay_loss', 'weighted_replay_loss',
              'current_grad_norm', 'replay_grad_norm', 'cosine', 'valid_signal')

    def __init__(self, config):
        self.config = config
        self.stage = None
        self._moments = None
        self._initialized = None
        self._records = []
        self._skipped = 0

    def start_stage(self, stage):
        if not isinstance(stage, int) or stage < 1:
            raise ValueError('GRB expects a positive chronological stage')
        self.stage = stage
        self._moments = None
        self._initialized = None
        self._records = []
        self._skipped = 0

    def combine(self, original_loss, current, forward, inverse, representation):
        if self.stage is None:
            raise RuntimeError('start_stage must precede GRB loss calls')
        if self.config.mode == 'original':
            return original_loss
        if current is None or forward is None or inverse is None or not torch.is_grad_enabled():
            self._skipped += 1
            return original_loss

        replay = forward + inverse  # inverse already includes Original mu
        moments = None
        if self.config.mode == 'fixed':
            weight = current.detach().new_tensor(self.config.fixed_weight)
            cosine = current.detach().new_zeros(())
            valid = current.detach().new_zeros((), dtype=torch.bool)
        else:
            if not representation.requires_grad or not current.requires_grad or not replay.requires_grad:
                self._skipped += 1
                return original_loss
            # The same forward/dropout/BN graph supplies both losses. Stop the
            # measurements at H: no extra RGCN/DGL backward or .grad accumulation.
            gc, = torch.autograd.grad(current, representation, retain_graph=True,
                                      create_graph=False, allow_unused=True)
            gr, = torch.autograd.grad(replay, representation, retain_graph=True,
                                      create_graph=False, allow_unused=True)
            if gc is None or gr is None:
                self._skipped += 1
                return original_loss
            with torch.no_grad():
                gc, gr = gc.detach().float(), gr.detach().float()
                moments = torch.stack((gc.square().sum(), gr.square().sum(), (gc * gr).sum()))
                # An invalid observation must not poison subsequent EMA values.
                good = torch.isfinite(moments).all() & (moments[:2] > 0).all()
                previous = self._moments if self._moments is not None else torch.zeros_like(moments)
                initialized = (self._initialized if self._initialized is not None
                               else moments.new_zeros((), dtype=torch.bool))
                smoothed = self.config.ema * previous + (1-self.config.ema) * moments
                smoothed = torch.where(initialized, smoothed, moments)
                self._moments = torch.where(good, smoothed, previous).detach()
                self._initialized = (initialized | good).detach()
                weight, cosine, valid = weight_from_moments(self._moments, self.config)
                valid = valid & good
                weight = torch.where(valid, weight, torch.ones_like(weight))
                cosine = torch.where(valid, cosine, torch.zeros_like(cosine))

        with torch.no_grad():
            norms = self._moments[:2].clamp_min(0).sqrt() if moments is not None else weight.new_zeros(2)
            record = torch.stack((weight.float(), current.detach().float(), replay.detach().float(),
                                  weight.float()*replay.detach().float(), norms[0], norms[1],
                                  cosine.float(), valid.float()))
            self._records.append(record.detach())
        if self.config.mode == 'fixed' and self.config.fixed_weight == 1:
            return original_loss  # exact bypass, including floating-point addition order
        # Preserve the two CE contributions, inverse expansion and original mu.
        return (torch.zeros_like(original_loss) + current) + weight*forward + weight*inverse

    def write_stage(self, run_dir):
        if self.stage is None:
            raise RuntimeError('No GRB stage to write')
        records = self._records
        self._records = []  # do not keep any stage tensors alive after writing
        summary = {'version': 1, 'stage': self.stage, 'mode': self.config.mode,
                   'measured_updates': len(records), 'skipped_updates': self._skipped,
                   'gradient_space': 'final_entity_representation',
                   'ema_resets_each_stage': True}
        if records:
            rows = torch.stack(records).cpu().tolist()  # one device transfer per stage
            summary['updates'] = [dict(zip(self.fields, row)) for row in rows]
            summary['mean_weight'] = sum(row[0] for row in rows) / len(rows)
            summary['min_weight'] = min(row[0] for row in rows)
            summary['max_weight'] = max(row[0] for row in rows)
            valid_rows = [row for row in rows if row[-1]]
            summary['negative_cosine_fraction'] = (
                sum(row[-2] < 0 for row in valid_rows) / len(valid_rows) if valid_rows else None)
        if run_dir is not None:
            # Original has no file writes here. Partial stages may leave an
            # extra record after interruption; completed_stage is authoritative.
            with (run_dir / 'grb-stages.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(summary, allow_nan=False) + '\n')
        return summary
