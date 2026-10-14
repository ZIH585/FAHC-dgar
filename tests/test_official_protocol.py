"""CPU regression against the author's pinned loop, not a rewritten reference.

The real chronological loop and real Adam/EarlyStopping run on scalar stub
models. No dataset, GPU, DGAR epoch, or checkpoint from an experiment is used.
Only upstream's unsupported Tensor.to(integer-GPU) calls are adapted to CPU.
"""
import ast
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import types
import unittest

import numpy as np
import torch

from rgcn.utils import EarlyStopping
from src.cli import build_arg_parser, validate_args
from src.runtime import (PROTOCOL, restore_evaluation_rng, restore_rng,
                         restore_training_state, rng_state, save_training_state)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'tests/reference/official_main.py.txt'


class CpuDevice(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                and len(node.args) == 1
                and ast.dump(node.args[0]) == ast.dump(ast.parse('args.gpu', mode='eval').body)):
            node.args[0] = ast.Constant('cpu')
        return node


def code_from(path, function, loop_only=False):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    if loop_only:
        branch = next(n for n in fn.body if isinstance(n, ast.If) and n.orelse
                      and any(isinstance(c, ast.Attribute) and c.attr == 'test'
                              for c in ast.walk(n.test)))
        tree = ast.Module(body=branch.orelse, type_ignores=[])
    else:
        tree = ast.Module(body=[fn], type_ignores=[])
    return compile(ast.fix_missing_locations(CpuDevice().visit(tree)), str(path), 'exec')


def draws():
    return (random.random(), float(np.random.random()), float(torch.rand(())))


class Interrupted(Exception):
    pass


def run_loop(path, directory, resume=False, interrupt_at=None, rter_mode='original'):
    """Execute the actual main.py chronological branch with tiny models."""
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    trace = []
    args = validate_args(build_arg_parser().parse_args([
        '--gpu', '-1', '--regcn_epochs', '3', '--diffu_epochs', '3',
        '--add-static-graph', 'false', '--run-dir', str(directory)]))
    args._provenance = {'protocol': PROTOCOL, 'synthetic': True}
    args.rter_mode = rter_mode
    args.resume = directory / 'latest.pt' if resume else None
    directory.mkdir(exist_ok=True)
    train = [np.array([[t, 0, (t + 1) % 8]], dtype=np.int64) for t in range(4)]
    valid = [np.array([[t, 0, (t + 2) % 8]], dtype=np.int64) for t in range(4)]
    test = [np.array([[t, 0, (t + 3) % 8]], dtype=np.int64) for t in range(4)]

    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(()))
            self.register_buffer('calls', torch.zeros((), dtype=torch.long))

        def get_loss(self, graphs, output, *a, **kw):
            trace.append(('predictor_update', int(output[0, 0]), float(self.weight.detach()),
                          int(optimizer.state.get(self.weight, {}).get('step', 0)), draws()))
            loss = (self.weight - float(output[0, 0])) ** 2
            return loss, loss * .3, loss * .1, self.weight.reshape(1, 1), loss * 0

    class Diffusion(Scalar):
        def gereration_feature(self, a, stage, generator, teacher, history, last,
                               static, query, output, cuda, train_list):
            if stage < 2:
                return output, None, None
            self.calls.add_(1)
            teacher.calls.add_(1)
            trace.append(('features', stage, tuple(query[0]), self.training, teacher.training,
                          float(self.weight.detach()), float(teacher.weight.detach()),
                          int(self.calls), int(teacher.calls), tuple(history or []), draws()))
            return output, torch.ones(1, 3, 2), torch.tensor([[0, 0, 1]])

        def forward(self, sequence, triples, a, train_flag, cuda, *rest, **kw):
            trace.append(('diffusion_update' if train_flag else 'diffusion_validation',
                          int(triples[0, 0]), float(self.weight.detach()), self.training,
                          int(optimizer_diffu.state.get(self.weight, {}).get('step', 0)), draws()))
            value = self.weight + torch.rand(())
            return None, value, None, None, None, value, None, value, None

        def loss_diffu_ce(self, rep, *a):
            return rep ** 2

        def diffu_rep_pre(self, rep, bank):
            return rep[0]

    # Keep the same four-model state in both loop executions. Mode setup outside
    # the loop is covered separately and by the real-module CPU smoke test.
    model, diffu, model_repre, model_diffpre = Scalar(), Diffusion(), Scalar(), Diffusion()
    for feature_model in (model_repre, model_diffpre):
        feature_model.requires_grad_(False).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.03, weight_decay=1e-5)
    optimizer_diffu = torch.optim.Adam(diffu.parameters(), lr=.03, weight_decay=1e-5)
    first_stage = 1
    if resume:
        first_stage, _ = restore_training_state(args, model, diffu, model_repre, model_diffpre,
                                               optimizer, optimizer_diffu, 'cpu')
    counts = {}

    def test_single(a, m, graphs, triples, stage, *rest):
        m.eval()
        is_test = torch.equal(triples, torch.from_numpy(test[stage]))
        key = ('test' if is_test else 'valid', stage)
        counts[key] = counts.get(key, 0) + 1
        # The first epoch is best; the last optimizer step must still survive.
        score = torch.tensor(1. / counts[key])
        trace.append(('test_single', key, float(m.weight.detach()), draws()))
        return score, [score] * 3, torch.ones(2)

    diff_validation_count = 0

    def get_rank(*a, **kw):
        nonlocal diff_validation_count
        score = torch.tensor(1. / (diff_validation_count % 3 + 1))
        diff_validation_count += 1
        return score, score, score, score

    def final_test(a, m, *pos, **kw):
        m.load_state_dict(torch.load(directory / 'model.pt', weights_only=True))
        m.eval()
        trace.append(('final_test', float(m.weight.detach()), float(model_diffpre.weight.detach()),
                      int(model_diffpre.calls), draws()))
        return (0., 0., 0., 0., kw['last_history'], kw['last_output'], [], [], [], [], [], [], [], [])

    def save(*a, **kw):
        save_training_state(*a, **kw)
        if interrupt_at == a[1]:
            raise Interrupted()

    ns = dict(args=args, model=model, diffu=diffu, model_repre=model_repre, model_diffpre=model_diffpre,
              optimizer=optimizer, optimizer_diffu=optimizer_diffu, train_list=train, valid_list=valid,
              test_list=test, first_stage=first_stage, use_cuda=False, run_device=torch.device('cpu'),
              num_nodes=8, num_rels=1, static_graph=None, static_graph_diffu=None, torch=torch,
              np=np, random=random, tqdm=lambda x: x, EarlyStopping=EarlyStopping,
              model_state_file=str(directory / 'model.pt'), diffu_model_state_file=str(directory / 'diffusion.pt'),
              all_ans_list_valid=[{}] * 4, all_ans_list_test=[{}] * 4,
              all_ans_list_r_test=[{}] * 4, all_ans_list_r_valid=[{}] * 4,
              build_sub_graph=lambda n, r, triple, *a: int(triple[0, 0]),
              getReverseSample=lambda triples, r: torch.cat((triples, triples)),
              test_single=test_single, test=final_test,
              fitlog=types.SimpleNamespace(finish=lambda: None),
              utils=types.SimpleNamespace(get_total_rank=get_rank,
                  stat_ranks_record=lambda rank, name: (rank, [rank] * 3)),
              save_training_state=save)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_from(path, 'run_experiment', loop_only=True), ns)
    except Interrupted:
        pass
    return trace, draws()


class OfficialProtocolTests(unittest.TestCase):
    def test_reference_is_exact_upstream_blob(self):
        manifest = json.loads((REFERENCE.parent / 'manifest.json').read_text())
        expected = next(f['sha'] for f in manifest['files'] if f['path'] == 'src/main.py')
        source = REFERENCE.read_bytes()
        self.assertEqual(hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest(), expected)

    def test_chronological_loop_matches_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, reference_rng = run_loop(REFERENCE, Path(tmp) / 'reference')
            actual, actual_rng = run_loop(ROOT / 'src/main.py', Path(tmp) / 'actual')
        self.assertEqual(actual, reference)
        self.assertEqual(actual_rng, reference_rng)
        steps = [row[3] for row in actual if row[0] == 'predictor_update']
        self.assertEqual(steps, list(range(9)))  # catches best-epoch Adam rollback

    def test_resume_matches_uninterrupted_and_repairs_partial_best_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            full, full_rng = run_loop(ROOT / 'src/main.py', Path(tmp) / 'full')
            run = Path(tmp) / 'interrupted'
            partial, _ = run_loop(ROOT / 'src/main.py', run, interrupt_at=2)
            torch.save({'invalid_partial_stage': torch.tensor(-1)}, run / 'model.pt')
            torch.save({'invalid_partial_stage': torch.tensor(-1)}, run / 'diffusion.pt')
            remainder, restored_rng = run_loop(ROOT / 'src/main.py', run, resume=True)
            self.assertEqual(full, partial + remainder)
            self.assertEqual(full_rng, restored_rng)

    def test_explicit_seed_and_no_method_options(self):
        parser = build_arg_parser()
        self.assertIsNone(validate_args(parser.parse_args([])).seed)
        self.assertEqual(validate_args(parser.parse_args(['--seed', '1'])).seed, 1)
        options = ' '.join(parser._option_string_actions)
        for name in ('fahc', 'ecri', 'qcdr', 'hcp-selection', 'kd-weight'):
            self.assertNotIn(name, options)

    def test_scientific_defaults_match_author_parser(self):
        tree = ast.parse(REFERENCE.read_text())
        entry = next(n for n in tree.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Compare))
        setup = []
        for node in entry.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'parse_args'):
                break
            setup.append(node)
        ns = {'argparse': argparse}
        exec(compile(ast.fix_missing_locations(ast.Module(body=setup, type_ignores=[])), str(REFERENCE), 'exec'), ns)
        official = vars(ns['parser'].parse_args([]))
        actual = vars(build_arg_parser().parse_args([]))
        # Upstream's unused seed flag is exposed as an explicit opt-in here.
        for key, value in official.items():
            if key != 'seed':
                self.assertEqual(actual[key], value, key)

    def test_standalone_test_restores_end_training_rng_and_rejects_stale_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            trace, _ = run_loop(ROOT / 'src/main.py', run)
            args = types.SimpleNamespace(run_dir=run, diffuc='add',
                _provenance={'protocol': PROTOCOL, 'synthetic': True})
            random.seed(111)
            np.random.seed(111)
            torch.manual_seed(111)
            restore_evaluation_rng(args, 'cpu', 3)
            self.assertEqual(draws(), trace[-1][-1])
            with self.assertRaises(ValueError):
                restore_evaluation_rng(args, 'cpu', 4)
            weights = torch.load(run / 'model.pt', weights_only=True)
            weights['weight'].add_(1)
            torch.save(weights, run / 'model.pt')
            with self.assertRaises(ValueError):
                restore_evaluation_rng(args, 'cpu', 3)

    def test_main_feature_generators_stay_in_train_mode(self):
        tree = ast.parse((ROOT / 'src/main.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name)
                 and node.func.value.id in ('model_repre', 'model_diffpre')
                 and node.func.attr in ('train', 'eval')]
        self.assertEqual(sorted((c.func.value.id, c.func.attr) for c in calls),
                         [('model_diffpre', 'train'), ('model_repre', 'train')])

    def test_final_evaluation_matches_upstream_history_order_rng_and_ranks(self):
        # Execute both real test() functions. Scoring is tiny, but history/HCP
        # sampling, ordering, argument passing and metric aggregation are real.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'data/SMALL/history_snap_v2'
            cache.mkdir(parents=True)
            for t in range(5):
                (cache / f'snap_{t}.pkl').write_bytes(pickle.dumps({'0': {(0, 0, t), (0, 0, 5 - t)}}))
            src = root / 'src'
            src.mkdir()
            args = validate_args(build_arg_parser().parse_args([
                '--gpu', '-1', '--data-root', str(root / 'data'), '--dataset', 'SMALL', '--reply_batch', '2']))

            def execute(path):
                trace = []
                random.seed(15)
                np.random.seed(15)
                torch.manual_seed(15)

                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.dummy = torch.nn.Parameter(torch.ones(()))

                    def predict(self, graphs, rels, static, triples, rep, *rest):
                        trace.append(('predict', tuple(graphs), draws()))
                        scores = torch.rand(2, 6)
                        return torch.cat((triples, triples)), scores, scores

                def generator(sequence, tag, a, *pos, **kw):
                    trace.append(('guide', kw['ct'], tuple(kw['history_glist']),
                                  tag.tolist(), draws()))
                    return None, torch.rand(len(tag), 3, 2), None, None, None, None, None, None, None

                def rank(triples, scores, answers, **kw):
                    ranks = 1 + scores.argsort(descending=True).argsort()[:, 0]
                    mrr = (1. / ranks).mean()
                    return mrr, mrr, ranks, ranks

                def record(ranks, *a):
                    return (1. / ranks).mean(), [(ranks <= k).float().mean() for k in (1, 3, 10)]

                ns = dict(torch=torch, np=np, random=random, pickle=pickle, tqdm=lambda x: x,
                          build_sub_graph=lambda n, r, triple, *a: int(triple[0, 2]),
                          utils=types.SimpleNamespace(get_total_rank=rank, stat_ranks_record=record,
                              stat_ranks=lambda ranks, name: record(torch.cat(ranks))[0]))
                exec(code_from(path, 'test'), ns)
                history = [np.array([[0, 0, i]]) for i in range(5)]
                queries = [np.array([[0, 0, 1]]) for _ in range(5)]
                previous = Path.cwd()
                try:
                    os.chdir(src)  # upstream's original relative HCP path
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = ns['test'](args, Model(), history, queries, 1, 6, False,
                            [{}] * 5, [{}] * 5, 'unused', None, 'valid', [1, 2, 3], [], None, generator)
                finally:
                    os.chdir(previous)
                return trace, tuple(float(v) for v in result[:4]), draws()

            self.assertEqual(execute(ROOT / 'src/main.py'), execute(REFERENCE))


if __name__ == '__main__':
    unittest.main()
