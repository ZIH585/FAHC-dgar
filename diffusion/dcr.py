"""Replay endpoints for DGAR; no learned state or changes to HCP selection."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


PROTOCOL = 'dcr-endpoint-v1'
MODES = ('original', 'mean', 'x0', 'guided-x0')


def active(args):
    # vl=True is the generator's original validation objective, not HCP replay.
    return getattr(args, 'dcr_mode', 'original') != 'original' and not args.vl


def release_guidance_graph(model):
    # RecurrentRGCN writes transient h/h_0 attributes during input guidance.
    # Its next forward always initializes them again. Do not retain a latent
    # autograd graph on the frozen guide between replay calls.
    if model is not None:
        for name in ('h', 'h_0'):
            value = getattr(model, name, None)
            if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
                setattr(model, name, value.detach())


def guided_tail(model, x0, base_tail, tags, history, static, cuda, args):
    """Partial derivative at base_tail with head/relation tokens held fixed.

    tags are historical TRAIN facts. Evaluation answers and last_output are
    deliberately absent. autograd.grad never accumulates parameter gradients.
    """
    if model is None or not history:
        raise ValueError('Guided DCR requires a predictor and nonempty historical graphs')
    try:
        with torch.enable_grad():
            tail = base_tail.detach().requires_grad_(True)
            features = torch.cat((x0[:, :2].detach(), tail.unsqueeze(1)), dim=1)
            states, _, relations, _, _ = model(history, static, cuda,
                features, tail, tags, diffu_rep=args.diffuc)
            entities = F.normalize(states[-1]) if model.layer_norm else states[-1]
            scores = model.decoder_ob(entities, relations, tags, mode='test')
            log_prob = F.log_softmax(scores, dim=-1)
            selected = log_prob.gather(1, tags[:, 2:3]).sum()
            gradient = torch.autograd.grad(selected, tail, create_graph=False)[0].detach()
        result = base_tail.detach() + args.classifier_scale * gradient
        return result.detach(), gradient
    finally:
        release_guidance_graph(model)


@torch.no_grad()
def sample(sampler, model, tags, item_rep, noise_x_t, condition, bank, mask_seq,
           history, last_output, static, args, cuda, mask=None, query_sub3=None):
    """Keep one denoiser call and the original highest diffusion timestep."""
    mode = args.dcr_mode
    if mode not in MODES[1:]:
        raise ValueError('DCR sampler expects an enabled endpoint mode')
    sampler._dcr_summary = None
    if len(tags) == 0:
        return item_rep.detach()
    t = tags.new_full((len(tags),), sampler.num_timesteps - 1)
    if mode == 'mean':
        # Reuse the author's exact predictor guidance at the posterior mean.
        # It is NOT guidance at x0; this is the noise-removal control.
        try:
            tail, _, x0 = sampler.p_mean_variance(item_rep, noise_x_t, model,
                tags, t, condition, bank, mask_seq, history, last_output,
                static, cuda, args, query_sub3)
            posterior = sampler.q_posterior_mean_variance(x0[:, -1], noise_x_t, t)
            scaled_gradient = tail.detach() - posterior.detach()
        finally:
            release_guidance_graph(model)
    else:
        x0 = sampler.xstart_model(item_rep.detach(), noise_x_t,
            sampler._scale_timesteps(t), condition, bank, mask_seq, query_sub3).detach()
        if mode == 'guided-x0':
            tail, gradient = guided_tail(model, x0, x0[:, -1], tags,
                history, static, cuda, args)
            scaled_gradient = args.classifier_scale * gradient
        else:
            tail = x0[:, -1]
            scaled_gradient = torch.zeros_like(tail)
    # Consume the original terminal draw, but do not add it. This prevents
    # simply deleting randn from shifting the shared stream for mean/guided-x0.
    # x0 deliberately omits guidance and its stochastic/BN operations.
    noise = torch.randn_like(noise_x_t)
    result = torch.cat((x0[:, :2].detach(), tail.detach().unsqueeze(1)), dim=1)
    sampler._dcr_summary = torch.stack((
        x0[:, 0].detach().norm(dim=-1).mean(),
        x0[:, -1].detach().norm(dim=-1).mean(),
        scaled_gradient.detach().norm(dim=-1).mean(),
        tail.detach().norm(dim=-1).mean(),
        noise.norm(dim=-1).mean())).detach()
    return result.detach()


class Recorder:
    """Aggregate detached norms on device; copy scalars only at stage boundaries."""
    def __init__(self, path):
        self.path = Path(path)
        self.key = None
        self.total = None
        self.calls = 0
        self.facts = 0
        self.coefficient = None

    def observe(self, stage, mode, guider_training, count, values, coefficient):
        key = (int(stage), mode, 'fit' if guider_training else 'evaluation')
        if self.key != key:
            self.flush()
            self.key = key
        values = values.detach() * count
        self.total = values.clone() if self.total is None else self.total + values
        self.calls += 1
        self.facts += count
        self.coefficient = float(coefficient)

    def flush(self):
        if self.total is None:
            return
        values = (self.total / self.facts).detach().cpu().tolist()
        names = ('x0_head_norm', 'x0_tail_norm', 'scaled_guidance_norm',
                 'returned_tail_norm', 'discarded_noise_norm')
        row = dict(protocol=PROTOCOL, stage=self.key[0], mode=self.key[1],
            phase=self.key[2], calls=self.calls, facts=self.facts,
            posterior_x0_coefficient=self.coefficient,
            norms=dict(zip(names, values)))
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.key = None
        self.total = None
        self.calls = 0
        self.facts = 0


def record(args, sampler, stage, model, count):
    recorder = getattr(args, '_dcr_recorder', None)
    summary = getattr(sampler, '_dcr_summary', None)
    if recorder is not None and summary is not None and count and stage is not None:
        recorder.observe(stage, args.dcr_mode, model.training if model is not None else False,
            count, summary, sampler.posterior_mean_coef1[-1])
    # Only detached scalar aggregates may persist, never replay tensors/graphs.
    sampler._dcr_summary = None
