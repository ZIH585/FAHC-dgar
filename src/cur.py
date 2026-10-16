"""CUR v1: train-only pre-update utility labels, query-only inference routing.

R = mean+Shared injection; 0 = no generated-feature injection. Both paths
share parameters and replay labels. The router never sees a target entity.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.runtime import cpu_copy, rng_state, restore_rng

DUAL = ("dual-replay", "dual-direction", "utility")


class SharedAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(width, rank, bias=False)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


class UtilityRouter(nn.Module):
    """Directional relation ID + 10 label-free numeric query features."""
    def __init__(self, relations, hidden):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.relation = nn.Embedding(2 * relations, 8)
            self.network = nn.Sequential(nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            # Initially use R on a tie. The last layer learns on the first step.
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, relations, features):
        return self.network(torch.cat((self.relation(relations), features), dim=1)).squeeze(-1)


def mean_sample(args, x0, mean, original):
    # Original noise has already been drawn; keep its RNG consumption.
    if getattr(args, "cur_mode", "original") == "original":
        return original
    return torch.cat((x0[:, :2], mean[:, None]), dim=1)


def chunks(count, size):
    """Avoid a final singleton: Original ConvTransE skips bn2 for N=1."""
    start = 0
    while start < count:
        end = min(count, start + size)
        if count - end == 1:
            end = count
        yield slice(start, end)
        start = end


@contextmanager
def isolated_predictor(model):
    state = rng_state(model.dynamic_emb.device)
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    missing = object()
    scratch = {name: model.__dict__.get(name, missing) for name in ("h", "h_0")}
    model.eval()
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in modes:
            module.training = training
        for name, value in scratch.items():
            if value is missing:
                model.__dict__.pop(name, None)
            else:
                model.__dict__[name] = value
        restore_rng(state, model.dynamic_emb.device)


def representations(model, history, static, features, tags):
    """Same parameters/history/features; change only feature injection."""
    result = []
    for inject in (True, False):
        states, _, relations, _, _ = model.forward(history, static,
            model.dynamic_emb.device.type == "cuda",
            model_output=features if inject else None,
            model_output_x_t=features[:, 2] if inject else None,
            tag=tags, diffu_rep=model.args.diffuc)
        embedding = F.normalize(states[-1]) if model.layer_norm else states[-1]
        result.append((embedding, relations))
    return result


def support_counts(model, history, tags):
    """Only sampled past HCP and the query's existing training history graph."""
    device = model.dynamic_emb.device
    entities = torch.zeros(model.num_ents, device=device)
    relations = torch.zeros(2 * model.num_rels, device=device)
    for graph in history:
        graph = graph.to(device)
        sources = graph.ndata["id"].flatten()[graph.edges()[0]]
        entities += torch.bincount(sources, minlength=model.num_ents)
        relations += torch.bincount(graph.edata["type"], minlength=2 * model.num_rels)
    hcp_entities = torch.bincount(torch.cat((tags[:, 0], tags[:, 2])), minlength=model.num_ents).float()
    hcp_relations = torch.bincount(tags[:, 1] % model.num_rels, minlength=model.num_rels).float().repeat(2)
    return tuple(torch.log1p(x) / torch.log1p(x.sum()).clamp_min(1)
                 for x in (entities, relations, hcp_entities, hcp_relations))


def query_features(scores_r, scores_0, known_queries, support, num_relations):
    """known_queries MUST be [subject, relation], never [subject, relation, target]."""
    if known_queries.ndim != 2 or known_queries.shape[1] != 2:
        raise ValueError("router inputs must contain only subject and relation")
    statistics = []
    top = []
    for scores in (scores_r, scores_0):
        probabilities = torch.softmax(scores.float(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(1) / max(1, math.log(scores.shape[1]))
        values = scores.topk(min(2, scores.shape[1]), dim=1).values
        gap = (values[:, 0] - values[:, -1]).tanh()
        statistics.extend((entropy, gap))
        top.append(scores.topk(min(10, scores.shape[1]), dim=1).indices)
    overlap = (top[0][:, :, None] == top[1][:, None, :]).any(2).float().mean(1)
    subject, relation = known_queries.unbind(1)
    entity_hist, relation_hist, entity_hcp, relation_hcp = support
    statistics.extend((overlap, entity_hist[subject], relation_hist[relation],
                       entity_hcp[subject], relation_hcp[relation], (relation >= num_relations).float()))
    return torch.stack(statistics, dim=1).detach()


def reciprocal_rank(scores, targets):
    """Full candidate set, raw midrank for ties; TRAIN labels only for fitting."""
    truth = scores.gather(1, targets[:, None])
    rank = 1 + (scores > truth).sum(1) + .5 * ((scores == truth).sum(1) - 1)
    return rank.float().reciprocal()


def route_choices(mode, probabilities, relations, num_relations, threshold):
    if mode == "dual-replay":
        return torch.ones_like(probabilities, dtype=torch.bool)
    if mode == "dual-direction":
        return relations >= num_relations
    return probabilities >= threshold


@torch.no_grad()
def paired_observations(model, history, static, triples, features, tags, with_scores=False):
    """Eval-only paired scores. Labels affect passive RR observations only."""
    pair = representations(model, history, static, features, tags)
    support = support_counts(model, history, tags)
    numeric, rr_r, rr_0, selected, probabilities, choices = [], [], [], [], [], []
    for section in chunks(len(triples), model.args.cur_score_batch_size):
        queries = triples[section]
        r, zero = [model.decoder_ob(e, rel, queries, mode="test") for e, rel in pair]
        x = query_features(r, zero, queries[:, :2], support, model.num_rels)
        p = model.cur_router(queries[:, 1], x).sigmoid()
        use_r = route_choices(model.args.cur_mode, p, queries[:, 1], model.num_rels, model.args.cur_threshold)
        if with_scores:
            selected.append(torch.where(use_r[:, None], r, zero))
        # Test/valid labels only enter these detached, passive diagnostics.
        rr_r.append(reciprocal_rank(r, queries[:, 2]))
        rr_0.append(reciprocal_rank(zero, queries[:, 2]))
        numeric.append(x)
        probabilities.append(p)
        choices.append(use_r)
    result = {"features": torch.cat(numeric), "rr_r": torch.cat(rr_r), "rr_0": torch.cat(rr_0),
              "probability": torch.cat(probabilities), "use_replay": torch.cat(choices)}
    if with_scores:
        result["scores"] = torch.cat(selected)
        # Entity routing does not redefine the separate relation prediction task.
        result["relation_scores"] = model.rdecoder(pair[0][0], pair[0][1], triples, mode="test")
    return result


def predict(model, history, static, triples, features, tags):
    observed = paired_observations(model, history, static, triples, features, tags, with_scores=True)
    model._cur_observation = {key: observed[key].detach() for key in ("rr_r", "rr_0", "probability", "use_replay")}
    return triples, observed["scores"], observed["relation_scores"]


def dual_loss(model, *args, **kwargs):
    replay = model.get_loss(*args, **kwargs)
    if kwargs.get("model_output") is None or not len(kwargs["model_output"]):
        return replay
    zero = model.get_loss(*args, **kwargs, feature_injection=False, entity_only=True)
    # Only entity CE is averaged. Preserve original replay-branch relation and
    # static objectives, diffusion target embeddings and their coefficients.
    return ((replay[0] + zero[0]) * .5, *replay[1:])


def utility_summary(observed, indices, choices):
    u = (observed["rr_r"] - observed["rr_0"])[indices]
    decision = choices[indices].float()
    weight = u.abs()
    scalars = {"queries": u.new_tensor(len(u)), "positive_fraction": (u > 0).float().mean(),
               "nonzero_fraction": (u != 0).float().mean(),
               "replay_raw_midrank_mrr": observed["rr_r"][indices].mean(),
               "no_injection_raw_midrank_mrr": observed["rr_0"][indices].mean(),
               "routed_raw_midrank_mrr": (observed["rr_0"][indices] + decision * u).mean(),
               "replay_fraction": decision.mean(),
               "gain_vs_replay": ((decision - 1) * u).mean(),
               "oracle_gain_vs_replay": (-u).clamp_min(0).mean(),
               "weighted_sign_accuracy": (((decision.bool() == (u > 0)) * weight).sum() / weight.sum().clamp_min(1e-12))}
    return dict(zip(scalars, torch.stack(list(scalars.values())).detach().cpu().tolist()))


class CURRuntime:
    """Small router Adam state; no replay buffer, teacher copy or probe graph."""
    def __init__(self, args, model):
        self.args, self.model, self.completed, self.stats = args, model, 0, {}
        self.optimizer = (torch.optim.Adam(model.cur_router.parameters(), lr=args.cur_router_lr)
                          if args.cur_mode in DUAL else None)

    def fit(self, stage, teacher, history, current_train, static, features, tags):
        if stage != self.completed + 1:
            raise ValueError("CUR stages must advance in chronological order")
        self.stats = {"event": "probe", "stage": stage, "mode": self.args.cur_mode}
        if self.optimizer is None or features is None or not len(features) or not len(current_train):
            self.stats["reason"] = "single_path_or_no_replay"
            return
        generator = torch.Generator().manual_seed(self.args.seed + 104729 * stage)
        indices = torch.randperm(len(current_train), generator=generator)[:self.args.cur_probe_facts].to(current_train.device)
        sampled = current_train[indices]
        queries = teacher.get_all_triple(sampled)
        # Main passes the just-restored previous-best predictor, before this
        # stage's task update. Extra eval restores buffers/modes/scratch/RNG.
        with isolated_predictor(teacher):
            observed = paired_observations(teacher, history, static, queries, features.detach(), tags)
        count = len(sampled)
        held = max(1, count // 4) if count >= 4 else 0
        train = torch.cat((torch.arange(count - held), torch.arange(count, 2 * count - held))).to(queries.device)
        holdout = torch.cat((torch.arange(count - held, count), torch.arange(2 * count - held, 2 * count))).to(queries.device)
        x, relations = observed["features"].detach(), queries[:, 1]
        utility = (observed["rr_r"] - observed["rr_0"]).detach()
        if not torch.isfinite(x).all() or not torch.isfinite(utility).all():
            raise ValueError("nonfinite CUR training features/utility")
        weights, labels = utility.abs(), (utility > 0).float()
        router = self.model.cur_router
        last_loss = None
        if bool(weights[train].sum() > 0):
            for _ in range(self.args.cur_router_steps):
                self.optimizer.zero_grad(set_to_none=True)
                logits = router(relations[train], x[train])
                last_loss = (weights[train] * F.binary_cross_entropy_with_logits(logits, labels[train], reduction="none")).sum() / weights[train].sum()
                last_loss.backward()
                self.optimizer.step()
        # Do not let router gradients enter the subsequent global norm clip.
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            probability = router(relations, x).sigmoid()
            learned = probability >= self.args.cur_threshold
        self.stats.update(train=utility_summary(observed, train, learned),
                          holdout=utility_summary(observed, holdout, learned) if held else None,
                          router_loss=None if last_loss is None else float(last_loss.detach().cpu()))

    def complete(self, stage):
        if stage != self.completed + 1:
            raise ValueError("nonconsecutive CUR completion")
        self.completed = stage
        if self.args.cur_diagnostics:
            from src.cur_diagnostics import append_json
            append_json(self.args.run_dir / "diagnostics" / "stages.jsonl", self.stats)

    def state_dict(self):
        return {"version": 1, "mode": self.args.cur_mode, "completed_stage": self.completed,
                "optimizer": None if self.optimizer is None else cpu_copy(self.optimizer.state_dict())}

    def load_state_dict(self, state, completed_stage):
        if (state.get("version") != 1 or state.get("mode") != self.args.cur_mode
                or state.get("completed_stage") != completed_stage):
            raise ValueError("CUR auxiliary state/version/stage mismatch")
        if (state.get("optimizer") is None) != (self.optimizer is None):
            raise ValueError("CUR router optimizer mismatch")
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.completed = completed_stage


def checkpoint_auxiliary(args, stage):
    runtime = args._cur_runtime
    runtime.complete(stage)
    return {"cur": runtime.state_dict()}
