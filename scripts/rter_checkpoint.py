"""Read-only loader for audited Original checkpoints; never restores optimizers."""
import json
from pathlib import Path

import numpy as np
import torch
from diffusion.model_21 import Att_Diffuse_model, create_model_diffu
from rgcn import utils
from rgcn.knowledge_graph import _read_triplets_as_list
from src.cli import build_arg_parser, validate_args
from src.rrgcn import RecurrentRGCN
from src.runtime import PROTOCOL, UPSTREAM_COMMIT, fingerprint, validate_history_cache

ROOT = Path(__file__).resolve().parents[1]
# Audited OFF snapshots; enabled checkpoints are never silently treated as Original.
BASELINE_SOURCES = {
    '6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538',
    'c22dc00022ff00ef8ebf9486d32f839ef9f60b5e0c59d04b79413401ba96b3d5',
    '313c60fe3724c443be5b48c83eb2de0e756730c6290f5e1fef4ecb6e514be455',
}


def load_original(options):
    run = options.run_dir.resolve()
    config = json.loads((run/'config.json').read_text())
    settings = config['settings']
    local_hash = fingerprint([p for f in ('src', 'diffusion', 'rgcn')
                              for p in (ROOT/f).glob('*.py')], ROOT)
    if (config['protocol'] != PROTOCOL or config['upstream_commit'] != UPSTREAM_COMMIT
            or config['source_sha256'] not in BASELINE_SOURCES | {local_hash}
            or any(settings.get(k, 'original') != 'original'
                   for k in ('grb_mode', 'dcr_mode', 'bcr_mode', 'rter_mode'))
            or any(k.startswith(('fahc', 'ecri', 'qcdr', 'hcp_selection')) for k in settings)
            or not settings['diffuc'] or settings['delete_feature_reply']
            or settings['delete_his_prompt'] or settings['delete_score'] or settings['vl']):
        raise ValueError('Use a completed audited Original checkpoint with all methods OFF')
    args = build_arg_parser().parse_args([])
    for key, value in settings.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.gpu, args.data_root = options.gpu, options.data_root.resolve()
    args.run_dir = None  # Never write to the input run.
    args = validate_args(args)
    cuda = args.gpu >= 0
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --gpu -1 only for CPU checks')
    device = torch.device(f'cuda:{args.gpu}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    data_dir = args.data_root/args.dataset
    paths = [data_dir/f'{x}.txt' for x in ('train', 'valid', 'test', 'stat')]
    if args.add_static_graph:
        paths.append(data_dir/'e-w-graph.txt')
    if (fingerprint(paths, data_dir) != config['data_sha256']
            or fingerprint(list((data_dir/'history_snap_v2').glob('snap_*.pkl')), data_dir)
            != config['history_sha256']):
        raise ValueError('Dataset/cache fingerprint differs from this checkpoint')
    saved = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
    if saved.get('version') != 2 or saved['provenance'] != config:
        raise ValueError('Checkpoint provenance mismatch')
    data = utils.load_data(args.dataset, data_root=args.data_root)
    snapshots = sum((utils.split_by_time(x) for x in (data.train, data.valid, data.test)), [])
    train, validation, test = utils.spilt_data(snapshots)
    if saved['completed_stage'] != len(train)-1:
        raise ValueError('Original training is incomplete')
    validate_history_cache(train, data.num_rels, data_dir/'history_snap_v2')
    args.num_nodes, args.num_rels = data.num_nodes, data.num_rels
    words, static_relations, static = 0, 0, None
    if args.add_static_graph:
        triples = np.asarray(_read_triplets_as_list(str(data_dir/'e-w-graph.txt'), {}, {}, load_time=False))
        static_relations, words = len(np.unique(triples[:, 1])), len(np.unique(triples[:, 2]))
        triples[:, 2] += data.num_nodes
        static = utils.build_sub_graph(data.num_nodes+words, static_relations, triples, cuda, args.gpu)
    model = RecurrentRGCN(args.decoder, args.encoder, data.num_nodes, data.num_rels, static_relations,
        words, args.n_hidden, args.attn_heads, args.opn, sequence_len=args.train_history_len,
        num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect,
        layer_norm=args.layer_norm, input_dropout=args.input_dropout,
        hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout,
        aggregation=args.aggregation, weight=args.weight, discount=args.discount,
        angle=args.angle, use_static=args.add_static_graph, entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction, use_cuda=cuda, gpu=args.gpu, args=args)
    generator = Att_Diffuse_model(create_model_diffu(args), args, args.encoder, data.num_nodes,
        data.num_rels, dropout=args.dropout, max_time=len(train), num_words=words,
        num_static_rels=static_relations, num_bases=args.n_bases)
    for filename, key, module in (('model.pt', 'model', model), ('diffusion.pt', 'diffusion', generator)):
        weights = torch.load(run/filename, map_location='cpu', weights_only=True)
        if weights.keys() != saved[key].keys() or any(not torch.equal(v, saved[key][k]) for k, v in weights.items()):
            raise ValueError(f'{filename} differs from completed latest.pt')
        if any(not torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
            raise ValueError(f'Nonfinite {filename}')
        module.load_state_dict(weights)
        module.to(device).requires_grad_(False)
    model.eval()
    generator.train()  # Upstream feature-generator mode is part of the baseline.
    rng = saved['rng']
    del saved
    return args, config, local_hash, train, validation, test, static, model, generator, device, cuda, rng
