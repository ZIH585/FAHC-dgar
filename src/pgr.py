"""PGR v1: training-only, stage-frozen predictive geometry of terminal noise.

The sensitivity matrix is a variance-weighted OUTER PRODUCT OF BATCH CE
GRADIENTS, not a per-query Fisher, Hessian, or measured forgetting score.
Nothing here selects HCP facts, alters the task loss, or reads held-out labels.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from src.runtime import cpu_copy, rng_state, restore_rng

MODES = ("original", "mean", "shared", "mean-shared", "geometry",
         "geometry-random", "geometry-isotropic")
ADAPTED = frozenset(MODES[2:])
GEOMETRY = frozenset(MODES[4:])


class SharedAdapter(nn.Module):
    """A(z)=z+U V z; U starts at zero, V at a nonzero random projection."""

    def __init__(self, width, rank):
        super().__init__()
        if not 1 <= rank <= width:
            raise ValueError("adapter rank must be in [1, width]")
        # Added after all Original parameters. Do not change later backbone or
        # diffusion initializations. Models are constructed on CPU in main.py.
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(width, rank, bias=False)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


def aggregate(tags, features, num_entities):
    """Original occurrence mean, plus conditional terminal-noise variance.

    variance is supplied separately by bank_variance; aggregation includes both
    positions even for a self-loop. Relation IDs do not index the entity bank.
    """
    ids = torch.cat((tags[:, 0], tags[:, 2]))
    counts = torch.bincount(ids, minlength=num_entities).to(features.dtype)
    bank = features.new_zeros(num_entities, features.shape[-1])
    bank.index_add_(0, ids, torch.cat((features[:, 0], features[:, 2])))
    bank = bank / counts.clamp_min(1).unsqueeze(1)
    return counts.nonzero().flatten(), bank


def bank_variance(tags, sigma, num_entities):
    """v_e=sum_{i:o_i=e} sigma_i^2/(n_head[e]+n_tail[e])^2."""
    if sigma.ndim != 2 or sigma.shape != (len(tags), 1):
        raise ValueError("terminal sigma must have shape [facts, 1]")
    counts = torch.bincount(torch.cat((tags[:, 0], tags[:, 2])),
                            minlength=num_entities).to(sigma.dtype)
    variance = sigma.new_zeros(num_entities)
    variance.index_add_(0, tags[:, 2], sigma[:, 0].square())
    return variance / counts.clamp_min(1).square()


def shrink_noise(noise, basis, weights, mode):
    """Row convention: epsilon S, S=I-Q diag(w) Q^T; isotropic equal energy."""
    if weights.numel() == 0:
        return noise
    if mode == "geometry-isotropic":
        energy = 1 - (2 * weights - weights.square()).sum() / noise.shape[-1]
        return noise * energy.clamp_min(0).sqrt()
    return noise - ((noise @ basis) * weights) @ basis.T


def geometry_from_gradients(gradients, variance, rank, strength, mode, seed):
    """No large Jacobian: form d x d geometry, then a rank-r contraction."""
    width = gradients[0].shape[1]
    matrix = gradients[0].new_zeros(width, width)
    for gradient in gradients:
        matrix.add_(gradient.T @ (variance[:, None] * gradient) / len(gradients))
    matrix = (matrix + matrix.T) * .5
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite training geometry")
    eigenvalues, vectors = torch.linalg.eigh(matrix.float())
    values = eigenvalues[-rank:].flip(0).clamp_min(0)
    basis = vectors[:, -rank:].flip(1)
    tau = eigenvalues.clamp_min(0).mean()
    weights = strength * values / (values + tau).clamp_min(torch.finfo(values.dtype).tiny)
    if mode == "geometry-random":
        # Private CPU RNG: no changes to the author's training/evaluation stream.
        generator = torch.Generator().manual_seed(seed)
        random_basis = torch.randn(width, rank, generator=generator)
        basis = torch.linalg.qr(random_basis, mode="reduced")[0].to(matrix.device)
    trace = eigenvalues.clamp_min(0).sum()
    if mode == "geometry-isotropic":
        predicted_after = trace * (1 - (2 * weights - weights.square()).sum() / width)
    else:
        projected = (basis * (matrix @ basis)).sum(dim=0)
        predicted_after = trace - ((2 * weights - weights.square()) * projected).sum()
    stats = {"eigenvalues": values.detach().cpu().tolist(),
             "trace": float(trace.detach().cpu()),
             "predicted_batch_ce_noise_proxy_after": float(predicted_after.clamp_min(0).detach().cpu()),
             "weights": weights.detach().cpu().tolist(),
             "retained_noise_energy": float((1 - (2 * weights - weights.square()).sum() / width).cpu())}
    return basis.to(gradients[0]), weights.to(gradients[0]), stats


@contextmanager
def isolated_predictor(model):
    """Only extra calibration is isolated. Original guidance stays TRAIN.

    Restore buffers, RNG, mixed submodule modes, instance overrides, and scratch
    attributes on success AND exception; never leave a calibration graph alive.
    """
    device = model.dynamic_emb.device
    state = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    sentinel = object()
    attributes = {name: model.__dict__.get(name, sentinel)
                  for name in ("select_entity", "h", "h_0")}
    model.eval()
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for module, training in modes:
            module.training = training
        for name, value in attributes.items():
            if value is sentinel:
                model.__dict__.pop(name, None)
            else:
                model.__dict__[name] = value
        restore_rng(state, device)


def pick_train_rows(rows, budget, seed):
    """Uniform private sampling; no new HCP selection or test query inputs."""
    if len(rows) <= budget:
        return rows
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=generator)[:budget].to(rows.device)
    return rows[indices]


def calibrate(model, tags, features, sigma, history, current_train, static, args, stage):
    """Pre-update previous-best model; only current train + sampled past HCP.

    Historical probes use the current training history window. This approximates
    compatibility with the upcoming task; it is NOT historical-query evaluation.
    Forward/inverse loss gradients are measured separately to avoid cancellation
    between directions (cancellation within a direction remains a limitation).
    """
    if not history or not len(tags) or not len(current_train):
        raise ValueError("calibration requires train history, train facts and HCP")
    seed = (args.seed or 0) + 104729 * stage
    current = pick_train_rows(current_train, args.pgr_geometry_facts, seed)
    past = pick_train_rows(tags, args.pgr_geometry_facts, seed + 1)
    # Both inputs are forward facts in the dataset or the existing cache. Cache
    # inverse rows are converted back before balancing query directions.
    past = past.clone()
    inverse = past[:, 1] >= model.num_rels
    past[inverse] = past[inverse][:, [2, 1, 0]]
    past[inverse, 1] -= model.num_rels
    probes = torch.cat((current, past))
    all_probes = model.get_all_triple(probes)
    ids, mean_bank = aggregate(tags, features.detach(), model.num_ents)
    variance = bank_variance(tags, sigma.detach(), model.num_ents)
    with isolated_predictor(model), torch.enable_grad():
        leaf = mean_bank.detach().requires_grad_(True)
        model.select_entity = lambda *a, **kw: (ids, model.pgr_adapter(leaf))
        states, _, relations, _, _ = model.forward(history, static,
            leaf.device.type == "cuda", features, features[:, 2], tags,
            diffu_rep=args.diffuc)
        embedding = F.normalize(states[-1]) if model.layer_norm else states[-1]
        logits = model.decoder_ob(embedding, relations, all_probes, mode="test")
        count = len(probes)
        losses = [F.cross_entropy(logits[:count], all_probes[:count, 2]),
                  F.cross_entropy(logits[count:], all_probes[count:, 2])]
        gradients = [torch.autograd.grad(loss, leaf, retain_graph=i == 0)[0].detach()
                     for i, loss in enumerate(losses)]
        basis, weights, stats = geometry_from_gradients(gradients, variance,
            args.pgr_risk_rank, args.pgr_risk_strength, args.pgr_mode, seed + 2)
        stats.update(train_current=len(current), train_past=len(past),
                     loss_forward=float(losses[0].detach().cpu()),
                     loss_inverse=float(losses[1].detach().cpu()),
                     gradient_norms=[float(g.norm().cpu()) for g in gradients])
    return {"basis": basis.detach().cpu(), "weights": weights.detach().cpu(), "stats": stats}


class PGRRuntime:
    """Detached CPU stage catalog; no train graphs or labels in checkpoints."""

    def __init__(self, args):
        self.args, self.catalog, self.context = args, {}, None
        self.stage, self._cached = None, None

    def begin_fit(self, stage, history, current_train, static):
        if self.context is not None or stage in self.catalog:
            raise ValueError("stage already fitting/completed")
        if sorted(self.catalog) != list(range(1, stage)):
            raise ValueError("PGR stages must follow chronological order")
        self.context = (stage, history, current_train, static)

    def end_fit(self):
        stage = self.context[0]
        if stage not in self.catalog:
            self.catalog[stage] = self.empty("no_training_replay")
        self.context, self._cached = None, None

    def empty(self, reason):
        return {"basis": torch.empty(self.args.n_hidden, 0), "weights": torch.empty(0),
                "stats": {"reason": reason}}

    @contextmanager
    def at_stage(self, stage):
        previous = self.stage
        self.stage = int(stage)
        try:
            yield
        finally:
            self.stage = previous

    def sample(self, model, tags, x0, mean, sigma, original):
        mode, stage = self.args.pgr_mode, self.stage
        if mode == "shared":
            return original
        if mode in ("mean", "mean-shared"):
            return torch.cat((x0[:, :2], mean[:, None]), dim=1)
        if mode not in GEOMETRY:
            raise ValueError("unexpected PGR sampling mode")
        if stage is None:
            raise ValueError("PGR generation missing time index")
        if stage not in self.catalog:
            if self.context is None or self.context[0] != stage:
                raise ValueError(f"No training-only PGR geometry for stage {stage}; resume its catalog")
            _, history, current, static = self.context
            features = torch.cat((x0[:, :2], mean[:, None]), dim=1).detach()
            self.catalog[stage] = calibrate(model, tags, features, sigma,
                history, current, static, self.args, stage)
            self._cached = None
        if self._cached is None or self._cached[0] != stage or self._cached[1] != mean.device:
            entry = self.catalog[stage]
            self._cached = (stage, mean.device, entry["basis"].to(mean), entry["weights"].to(mean))
        _, _, basis, weights = self._cached
        if weights.numel() == 0 or self.args.pgr_risk_strength == 0:
            return original
        noise = original[:, 2] - mean
        tail = mean + shrink_noise(noise, basis, weights, mode)
        return torch.cat((x0[:, :2], tail[:, None]), dim=1)

    def state_dict(self):
        return {"version": 1, "mode": self.args.pgr_mode, "catalog": cpu_copy(self.catalog)}

    def load_state_dict(self, state, completed_stage):
        if (state.get("version") != 1 or state.get("mode") != self.args.pgr_mode
                or sorted(state.get("catalog", {})) != list(range(1, completed_stage + 1))):
            raise ValueError("PGR checkpoint catalog/version/mode mismatch")
        catalog = cpu_copy(state["catalog"])
        for entry in catalog.values():
            q, w = entry["basis"], entry["weights"]
            if (q.shape != (self.args.n_hidden, len(w)) or w.ndim != 1
                    or len(w) not in (0, self.args.pgr_risk_rank)
                    or not torch.isfinite(q).all() or not torch.isfinite(w).all()
                    or (w < 0).any() or (w > 1).any()):
                raise ValueError("invalid PGR projector")
            if len(w) and not torch.allclose(q.T @ q, torch.eye(len(w)), atol=2e-5):
                raise ValueError("PGR basis is not orthonormal")
        self.catalog, self._cached = catalog, None


def checkpoint_auxiliary(args):
    runtime = getattr(args, "_pgr_runtime", None)
    return None if runtime is None else {"pgr": runtime.state_dict()}
