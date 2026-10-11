#!/usr/bin/env python3
"""Paired VALIDATION fusion counterfactuals from explicit existing checkpoints.

No optimizer, training, cache generation, or TEST labels are used for scoring.
A final checkpoint has seen later training snapshots: this is retrospective
mechanism analysis, NOT online forecasting or temporal representation drift.
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fahc_run_config import parser_without_models, fingerprint
from analyze_dgar_evolution import read_snapshots, js_divergence, sha256
from src.fahc_config import validate_args


def prepare():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, help="Optional launcher config.json")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA index; -1 for CPU")
    parser.add_argument("--dataset", default="ICEWS14s")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--snapshots", default="60,90,120,180,240,300,360")
    parser.add_argument("--snapshot-radius", type=int, default=0,
                        help="Expand each requested center to fixed neighboring days")
    parser.add_argument("--alphas", default="checkpoint,0.25,0.5,0.75,1.0")
    parser.add_argument("--evolution-window", type=int, default=30)
    parser.add_argument("--minimum-support", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--evolution-sample-cap", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    defaults = parser_without_models().parse_args([])
    if cli.run_config:
        source = json.loads(cli.run_config.read_text())
        settings = source.get("settings", source)
        if settings.get("dataset", cli.dataset) != cli.dataset:
            raise ValueError("Run config dataset differs from --dataset")
        for key, value in settings.items():
            if not hasattr(defaults, key):
                raise ValueError(f"Unknown run-config setting: {key}")
            setattr(defaults, key, value)
    defaults.dataset, defaults.gpu, defaults.seed = cli.dataset, cli.gpu, cli.seed
    defaults.vl = False  # Preserve final main.test decoder-guided HCP generation.
    validate_args(defaults)
    if defaults.diffuc != "add" or defaults.delete_feature_reply:
        raise ValueError("This alpha diagnostic requires diffuc=add and HCP fusion enabled")
    if (cli.gpu < -1 or cli.seed < 0 or cli.snapshot_radius < 0 or cli.evolution_window < 1 or cli.minimum_support < 1
            or cli.permutations < 1 or cli.evolution_sample_cap < cli.minimum_support):
        raise ValueError("Invalid GPU, seed, window, or support")
    centers = sorted(set(map(int, cli.snapshots.split(","))))
    alpha_tokens = list(dict.fromkeys(cli.alphas.split(",")))
    if "checkpoint" not in alpha_tokens:
        raise ValueError("Include checkpoint alpha as the paired reference")
    alpha_tokens = ["checkpoint"] + [x for x in alpha_tokens if x != "checkpoint"]
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1
           for x in alpha_tokens if x != "checkpoint"):
        raise ValueError("Counterfactual alphas must be finite and in [0, 1]")
    base = ROOT / "data" / cli.dataset
    files = [base / f"{name}.txt" for name in ("train", "valid", "test", "stat")]
    if defaults.add_static_graph:
        files.append(base / "e-w-graph.txt")
    for path in [cli.model_checkpoint, cli.diffusion_checkpoint, *files]:
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = [snap for name in ("train", "valid", "test")
           for _, snap in read_snapshots(base / f"{name}.txt")]
    train = [snap[:int(.8 * len(snap))] for snap in raw]
    valid = [snap[int(.8 * len(snap)):int(.9 * len(snap))] for snap in raw]
    if not centers or min(centers) < 2 or max(centers) >= len(raw):
        raise ValueError(f"Snapshots must be in 2..{len(raw) - 1}")
    center_to_snapshots = {str(center): list(range(max(2, center - cli.snapshot_radius),
        min(len(raw), center + cli.snapshot_radius + 1))) for center in centers}
    indices = sorted({index for window in center_to_snapshots.values() for index in window})
    cache = [base / "history_snap_v2" / f"snap_{i}.pkl" for i in range(len(raw))]
    for path in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; existing outputs are not overwritten")
    metadata = {"protocol": "retrospective_paired_validation_fusion_v1",
                "model_checkpoint": str(cli.model_checkpoint.resolve()),
                "diffusion_checkpoint": str(cli.diffusion_checkpoint.resolve()),
                "model_sha256": sha256(cli.model_checkpoint),
                "diffusion_sha256": sha256(cli.diffusion_checkpoint),
                "run_config": str(cli.run_config.resolve()) if cli.run_config else None,
                "run_config_sha256": sha256(cli.run_config) if cli.run_config else None,
                "settings": vars(defaults), "snapshots": indices, "alphas": alpha_tokens,
                "snapshot_centers": centers, "snapshot_radius": cli.snapshot_radius,
                "center_to_snapshots": center_to_snapshots,
                "validation_fact_count_by_center": {center: sum(len(valid[t]) for t in window)
                    for center, window in center_to_snapshots.items()},
                "data_sha256": fingerprint(files), "history_sha256": fingerprint(cache),
                "evolution_window": cli.evolution_window, "minimum_support": cli.minimum_support,
                "permutations": cli.permutations, "evolution_sample_cap": cli.evolution_sample_cap,
                "dry_run_validation_scope": "files/config/hashes_only; no checkpoint tensor loading",
                "caveat": "Final checkpoint may have seen future TRAIN facts; validation only; no online claim",
                "filter": "same_snapshot_validation_answers_only",
                "top10": "unfiltered candidate overlap against checkpoint-alpha predictions",
                "randomness": "fixed seed per snapshot; does not reproduce main.test continuous RNG sequence",
                "fact_clusters": "fact_cluster_id is snapshot:index; forward/inverse share one original validation fact cluster",
                "evolution": "past-training marginal tail JS; subject composition/sampling can confound it",
                "entity_evolution": "Known query SUBJECT only; in/out relation-role JS and direction+relation+neighbor churn; balanced permutation null",
                "representation": "fusion perturbation of same checkpoint and history; NOT temporal drift"}
    if cli.run_config:
        for key in ("data_sha256", "history_sha256"):
            if key in source and source[key] != metadata[key]:
                raise ValueError(f"Run config {key} differs from the current data/cache")
        metadata["reported_training_seed"] = settings.get("seed")
    metadata["source_sha256"] = fingerprint([p for directory in ("src", "diffusion", "rgcn")
        for p in (ROOT / directory).rglob("*.py")] + [Path(__file__),
        ROOT / "scripts/fahc_run_config.py", ROOT / "scripts/analyze_dgar_evolution.py"])
    return cli, defaults, base, train, valid, indices, alpha_tokens, metadata


def prefix_features(train, t, window, num_rels, minimum):
    tails = [defaultdict(Counter), defaultdict(Counter)]
    frequencies, relation_frequency, last_seen = Counter(), Counter(), {}
    for i, snap in enumerate(train[:t]):
        for subject, relation, object_ in snap:
            frequencies.update((subject, object_)); relation_frequency[relation] += 1
            last_seen[subject] = last_seen[object_] = i
            band = 1 if i >= t - window else 0 if i >= t - 2 * window else None
            if band is not None:
                tails[band][relation][object_] += 1
                tails[band][relation + num_rels][subject] += 1
    evolution = {}
    for relation in tails[0].keys() | tails[1].keys():
        a, b = tails[0][relation], tails[1][relation]
        supported = min(sum(a.values()), sum(b.values())) >= minimum
        evolution[relation] = (js_divergence(a, b) if supported else None,
                               sum(a.values()), sum(b.values()))
    return frequencies, relation_frequency, last_seen, evolution


def subject_evolution(train, t, subjects, window, minimum, cap, permutations, seed):
    """Past TRAIN only. Balanced event samples and within-subject nulls."""
    events = [defaultdict(list), defaultdict(list)]
    for i in range(max(0, t - 2 * window), t):
        band = int(i >= t - window)
        for head, relation, tail in train[i]:
            if head in subjects:
                events[band][head].append((0, relation, tail))
            if tail in subjects:
                events[band][tail].append((1, relation, head))
    def comparison(left, right):
        role_js = js_divergence(Counter((d, r) for d, r, _ in left),
                                Counter((d, r) for d, r, _ in right))
        a, b = set(left), set(right)
        churn = 1 - len(a & b) / len(a | b) if a or b else None
        return role_js, churn
    result = {}
    for subject in sorted(subjects):
        old, recent = events[0][subject], events[1][subject]
        count = min(cap, len(old), len(recent))
        fields = {"prefix_subject_old_event_count": len(old),
                  "prefix_subject_recent_event_count": len(recent),
                  "prefix_subject_balanced_sample_count": count,
                  "prefix_subject_evolution_reliable": count >= minimum,
                  "prefix_subject_neighbor_churn_full": comparison(old, recent)[1]}
        for name in ("role_JS", "neighbor_churn"):
            for suffix in ("", "_null_mean", "_excess", "_permutation_p"):
                fields[f"prefix_subject_{name}{suffix}"] = None
        if count >= minimum:
            rng = random.Random(seed + 1000003 * t + 9176 * subject)
            left, right = rng.sample(old, count), rng.sample(recent, count)
            observed = comparison(left, right)
            pooled, null = left + right, [[], []]
            for _ in range(permutations):
                rng.shuffle(pooled)
                values = comparison(pooled[:count], pooled[count:])
                for index, value in enumerate(values):
                    null[index].append(value)
            for index, name in enumerate(("role_JS", "neighbor_churn")):
                null_mean = statistics.mean(null[index]); value = observed[index]
                fields[f"prefix_subject_{name}"] = value
                fields[f"prefix_subject_{name}_null_mean"] = null_mean
                fields[f"prefix_subject_{name}_excess"] = value - null_mean
                fields[f"prefix_subject_{name}_permutation_p"] = (1 + sum(x >= value for x in null[index])) / (permutations + 1)
        result[subject] = fields
    return result


def construct(args, base, snapshots, device):
    import numpy as np
    import torch
    from rgcn.utils import build_sub_graph
    from src.rrgcn import RecurrentRGCN
    from diffusion.model_21 import create_model_diffu, Att_Diffuse_model
    num_nodes, num_rels = map(int, (base / "stat.txt").read_text().split()[:2])
    args.num_nodes, args.num_rels = num_nodes, num_rels
    n_static, n_words, graph, diff_graph = 0, 0, None, None
    if args.add_static_graph:
        triples = np.loadtxt(base / "e-w-graph.txt", dtype=np.int64).reshape(-1, 3)
        n_static, n_words = len(np.unique(triples[:, 1])), len(np.unique(triples[:, 2]))
        triples[:, 2] += num_nodes
        graph = build_sub_graph(num_nodes + n_words, n_static, triples, device.type == "cuda", args.gpu)
        diff_graph = build_sub_graph(num_nodes + n_words, n_static, triples, device.type == "cuda", args.gpu)
    model = RecurrentRGCN(args.decoder, args.encoder, num_nodes, num_rels, n_static,
        n_words, args.n_hidden, args.attn_heads, args.opn, sequence_len=args.train_history_len,
        num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
        layer_norm=args.layer_norm, input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
        aggregation=args.aggregation, weight=args.weight, discount=args.discount,
        angle=args.angle, use_static=args.add_static_graph, entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction, use_cuda=device.type == "cuda",
        gpu=args.gpu, analysis=False, args=args).to(device)
    diffusion = Att_Diffuse_model(create_model_diffu(args), args, args.encoder,
        num_nodes, num_rels, dropout=args.dropout, max_time=snapshots, num_words=n_words,
        num_static_rels=n_static, num_bases=args.n_bases).to(device)
    return model, diffusion, graph, diff_graph, num_nodes, num_rels


def metrics(ranks):
    return {"MRR": statistics.mean(1 / r for r in ranks),
            **{f"Hits@{k}": statistics.mean(r <= k for r in ranks) for k in (1, 3, 10)}}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(cli, args, base, train, valid, indices, tokens, metadata):
    print(json.dumps({"event": "checkpoint_source", "model": metadata["model_checkpoint"],
                      "diffusion": metadata["diffusion_checkpoint"], "run_config": metadata["run_config"],
                      "protocol": metadata["protocol"], "caveat": metadata["caveat"]}), flush=True)
    import numpy as np
    import torch
    from rgcn.utils import build_sub_graph, load_all_answers_for_filter, get_total_rank
    from diffusion.fahc_selector import load_hcp_candidates
    from src.fahc_runtime import isolated_rng
    device = torch.device(f"cuda:{cli.gpu}" if cli.gpu >= 0 else "cpu")
    if cli.gpu >= 0:
        if not torch.cuda.is_available() or cli.gpu >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable")
        torch.cuda.set_device(cli.gpu)
    with isolated_rng(cli.seed, device):
        model, diffusion, static, static_diffu, n, nr = construct(args, base, len(train), device)
    for module, path in ((model, cli.model_checkpoint), (diffusion, cli.diffusion_checkpoint)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=True); module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    checkpoint_alpha = model.alpha.detach().clone()
    if not bool(torch.isfinite(checkpoint_alpha).all()):
        raise ValueError("Nonfinite checkpoint alpha")
    metadata["checkpoint_alpha"] = float(checkpoint_alpha)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    (cli.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records, summaries = [], []
    def graphs(t):
        return [build_sub_graph(n, nr, np.asarray(snap), device.type == "cuda", cli.gpu)
                for snap in train[max(0, t - args.test_history_len):t]]
    try:
        for t in indices:
            if not valid[t]:
                raise ValueError(f"Empty validation snapshot: {t}")
            queries = torch.tensor(valid[t], dtype=torch.long, device=device)
            history, previous = graphs(t), graphs(t - 1)
            rng = random.Random(cli.seed + 7000003 + 1000003 * t)
            times = list(range(t)) if t - 1 < args.reply_batch else rng.sample(range(t), args.reply_batch)
            candidates = load_hcp_candidates(cli.dataset, times, queries[:, 0].cpu().tolist(), base / "history_snap_v2")
            facts = torch.tensor(sorted({x for p in candidates for x in p.triples}),
                                 dtype=torch.long, device=device).reshape(-1, 3)
            if not len(facts):
                raise ValueError(f"No HCP facts at snapshot {t}; fusion diagnostic is uninformative")
            with isolated_rng(cli.seed + 7000003 + 1000003 * t, device), torch.no_grad():
                model.alpha.copy_(checkpoint_alpha)
                _, feature, *_ = diffusion(None, facts, args, False, device.type == "cuda",
                    static_graph=static_diffu, ct=t, model=model, history_glist=previous,
                    triples=torch.tensor(train[t - 1], dtype=torch.long, device=device))
            feature = feature.detach(); objects = feature[:, -1, :]
            if not bool(torch.isfinite(feature).all()):
                raise ValueError(f"Nonfinite HCP feature at snapshot {t}")
            entities, mean_hcp = model.select_entity(facts, feature, device.type == "cuda")
            entity_ids = entities.cpu().tolist()
            answers = load_all_answers_for_filter(valid[t], nr)
            frequency, rel_frequency, last_seen, evolution = prefix_features(train, t, cli.evolution_window, nr, cli.minimum_support)
            # Inverse queries know their own subject; no true target is a gate feature.
            known_subjects = {s for s, _, _ in valid[t]} | {o for _, _, o in valid[t]}
            subject_features = subject_evolution(train, t, known_subjects, cli.evolution_window,
                cli.minimum_support, cli.evolution_sample_cap, cli.permutations, cli.seed)
            baseline_ranks = baseline_top = baseline_rep = None
            for token in tokens:
                norm_ratio, hcp_cosine = [], []
                original_forward = model.rgcn.forward
                def capture(*positional, **keywords):
                    output = original_forward(*positional, **keywords)
                    current = torch.nn.functional.normalize(output, dim=-1) if args.layer_norm else output
                    norm_ratio[:] = (mean_hcp[entities].norm(dim=-1) / current[entities].norm(dim=-1).clamp_min(1e-8)).cpu().tolist()
                    hcp_cosine[:] = torch.nn.functional.cosine_similarity(mean_hcp[entities], current[entities], dim=-1).cpu().tolist()
                    return output
                # DGAR explicitly calls rgcn.forward(), bypassing PyTorch module hooks.
                model.rgcn.forward = capture
                try:
                    with isolated_rng(cli.seed + 7000003 + 1000003 * t, device), torch.no_grad():
                        if token == "checkpoint":
                            model.alpha.copy_(checkpoint_alpha)
                        else:
                            model.alpha.fill_(float(token))
                        triples, score, _ = model.predict(history, nr, static, queries,
                            feature, objects, facts, device.type == "cuda")
                        if not bool(torch.isfinite(score).all()):
                            raise ValueError(f"Nonfinite score at snapshot {t}, alpha {token}")
                        top = score.topk(min(10, n), dim=-1).indices.cpu().tolist()
                        _, _, raw_rank, rank = get_total_rank(triples, score.clone(), answers, 1000)
                        ranks, raw_ranks = rank.cpu().tolist(), raw_rank.cpu().tolist()
                        rep = (torch.nn.functional.normalize(model.h, dim=-1) if args.layer_norm else model.h).detach().cpu()
                finally:
                    model.rgcn.forward = original_forward
                    model.alpha.copy_(checkpoint_alpha)
                if token == "checkpoint":
                    baseline_ranks, baseline_top, baseline_rep = ranks, top, rep.clone()
                if not norm_ratio or any(not math.isfinite(x) for x in norm_ratio + hcp_cosine):
                    raise ValueError(f"Missing/nonfinite fusion statistics at snapshot {t}")
                overlap = [len(set(a) & set(b)) / len(a) for a, b in zip(top, baseline_top)]
                ratio_by_entity, cosine_by_entity = dict(zip(entity_ids, norm_ratio)), dict(zip(entity_ids, hcp_cosine))
                perturbation = (rep - baseline_rep).norm(dim=-1)
                summary = {"snapshot": t, "alpha_label": token, "alpha": float(checkpoint_alpha) if token == "checkpoint" else float(token),
                    "query_count": len(ranks), "hcp_fact_count": len(facts),
                    "hcp_time_indices": json.dumps(times),
                    "hcp_inverse_fact_count": int((facts[:, 1] >= nr).sum()), **metrics(ranks),
                    "top10_overlap": statistics.mean(overlap), "HCP_current_norm_ratio_median": statistics.median(norm_ratio),
                    "entity_mean_l2_fusion_perturbation": float(perturbation.mean())}
                summaries.append(summary); print(json.dumps(summary), flush=True)
                for i, (subject, relation, object_) in enumerate(triples.cpu().tolist()):
                    js, old_count, recent_count = evolution.get(relation, (None, 0, 0))
                    records.append({"snapshot": t, "alpha_label": token, "subject": subject, "relation": relation,
                        "object": object_, "direction": "forward" if relation < nr else "inverse",
                        "fact_cluster_id": f"{t}:{i % len(queries)}", "raw_rank": raw_ranks[i],
                        "reciprocal_rank": 1 / ranks[i],
                        **{f"hit{k}": ranks[i] <= k for k in (1, 3, 10)},
                        "rank": ranks[i], "reference_rank": baseline_ranks[i], "rank_change": ranks[i] - baseline_ranks[i],
                        "reciprocal_rank_change": 1 / ranks[i] - 1 / baseline_ranks[i], "top10_overlap": overlap[i],
                        "top10_entities": json.dumps(top[i]), "subject_fusion_l2": float(perturbation[subject]),
                        "hcp_subject_touched": subject in ratio_by_entity,
                        "subject_HCP_current_norm_ratio": ratio_by_entity.get(subject),
                        "subject_HCP_current_cosine": cosine_by_entity.get(subject),
                        "target_candidate_fusion_l2": float(perturbation[object_]),
                        "prefix_subject_frequency": frequency[subject], "prefix_target_frequency": frequency[object_],
                        "prefix_relation_frequency": rel_frequency[relation % nr],
                        "subject_recency": t - last_seen[subject] if subject in last_seen else None,
                        "prefix_tail_JS": js, "old_window_count": old_count, "recent_window_count": recent_count,
                        **subject_features[subject]})
    finally:
        model.alpha.copy_(checkpoint_alpha)
    write_csv(cli.output_dir / "snapshot_metrics.csv", summaries)
    write_csv(cli.output_dir / "query_metrics.csv", records)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["alpha_label"], row["relation"]].append(row["rank"])
    relation_rows = [{"alpha_label": a, "relation": r, "query_count": len(values), **metrics(values)}
                     for (a, r), values in grouped.items()]
    write_csv(cli.output_dir / "relation_metrics.csv", relation_rows)
    (cli.output_dir / "status.json").write_text(json.dumps({"complete": True, "snapshots": indices}) + "\n")


if __name__ == "__main__":
    prepared = prepare()
    if prepared[0].dry_run:
        print(json.dumps({"validated": True, **prepared[-1]}, indent=2))
    else:
        evaluate(*prepared)
