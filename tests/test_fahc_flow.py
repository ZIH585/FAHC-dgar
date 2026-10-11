"""Tiny CPU control-flow checks; every optimizer step is a state-only mock."""

import copy
import io
import json
import os
import pickle
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import src.main as main
from diffusion import fahc_selector


def _snapshot(time):
    return np.array([[0, i % 2, i + 1, time] for i in range(10)], dtype=np.int64)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def get_loss(self, *args, **kwargs):
        entity = self.weight.square()
        relation = 0.1 * self.weight.square()
        static = 0 * self.weight
        return entity, relation, static, self.weight.reshape(1, 1), static


class _Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.feature_calls = []

    def gereration_feature(self, *args, **kwargs):
        time_index, output = args[1], args[8]
        facts = kwargs.get("hcp_triples")
        self.feature_calls.append((time_index, facts))
        if not facts:
            return output, None, None
        replay = torch.as_tensor(facts, dtype=torch.long).reshape(-1, 3)
        return output, torch.ones((len(replay), 3, 8)), replay

    def forward(self, sequence, triples, *args, **kwargs):
        representation = self.weight.reshape(1, 1, 1).expand(len(triples), 3, 8)
        return (None, representation, None, None, None,
                torch.zeros((12, 8)), None, None, None)

    def loss_diffu_ce(self, *args, **kwargs):
        return self.weight.square()

    def diffu_rep_pre(self, representations, entity_embeddings):
        return torch.zeros((len(representations[0]), 12))


class FAHCFlowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.original_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.original_cwd)
        root = Path(self.temporary.name)
        (root / "src").mkdir()
        os.chdir(root / "src")
        self.history_dir = root / "data" / "Tiny" / "history_snap_v2"
        self.history_dir.mkdir(parents=True)
        self.snapshots = [_snapshot(time) for time in range(3)]
        for index, snapshot in enumerate(self.snapshots):
            history = {"0": {tuple(map(int, row[:3])) for row in snapshot[:8]}}
            with (self.history_dir / f"snap_{index}.pkl").open("wb") as stream:
                pickle.dump(history, stream)
        self.log_path = root / "fahc.jsonl"

    def _args(self, mode="risk+coverage", diffusion="add", diagnostics=False,
              start_reply=0):
        return SimpleNamespace(
            seed=5, gpu=-1, dataset="Tiny", test=False,
            hcp_selection=mode, fahc_diagnostics=diagnostics,
            fahc_kd_weight=0.0, fahc_log=str(self.log_path),
            encoder="fake", decoder="fake", n_layers=1, dilate_len=1,
            train_history_len=2, test_history_len=2,
            weight=0.5, discount=1, angle=10, dropout=0.0,
            input_dropout=0.0, hidden_dropout=0.0, feat_dropout=0.0,
            n_hidden=8, n_bases=1, n_basis=1, attn_heads=2, opn="sub",
            self_loop=False, skip_connect=False, layer_norm=False,
            aggregation="none", entity_prediction=True,
            relation_prediction=True, run_analysis=False,
            add_static_graph=False, lr=0.01, n_epochs=1,
            regcn_epochs=2, diffu_epochs=1, accumulation_steps=1,
            reply_batch=1, fahc_fact_budget=2,
            fahc_candidate_multiplier=2.0, fahc_probe_facts=2,
            fahc_probe_lr=None, fahc_probe_steps=1,
            fahc_score_batch_size=8, task_weight=0.5, fahc_eta=1.0,
            fahc_time_bin_size=1, fahc_entity_weight=1.0,
            grad_norm=1.0, fahc_kd_temperature=2.0,
            fahc_kd_confidence=0.5, fahc_eval_every=0,
            fahc_past_eval_samples=0, delete_feature_reply=False,
            diffuc=diffusion, start_reply=start_reply, vl=False,
        )

    def _run(self, args, *, stop_after_first_epoch=False):
        checkpoint_store = {}
        load_paths = []
        features = []
        probe_observations = []
        selection_observations = []
        materialize_observations = []
        final_evaluations = []

        class FakeEarlyStopping:
            def __init__(self, patience, verbose, path):
                self.path = path
                self.best_score = None
                self.delta = 0.0
                self.early_stop = False

            def __call__(self, mrr, model):
                score = float(mrr)
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    checkpoint_store[self.path] = copy.deepcopy(model.state_dict())
                else:
                    self.early_stop = True
                if stop_after_first_epoch and "diffu_models" not in self.path:
                    self.early_stop = True

        def fake_step(optimizer, closure=None):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    count = float(state["step"]) + 1 if state else 1.0
                    state["step"] = torch.tensor(count)
                    state["exp_avg"] = torch.full_like(parameter, count)
                    state["exp_avg_sq"] = torch.full_like(parameter, count)

        def fake_load(path, map_location=None):
            load_paths.append(str(path))
            if str(path) not in checkpoint_store:
                raise FileNotFoundError(str(path))
            return copy.deepcopy(checkpoint_store[str(path)])

        def fake_feature(*args, **kwargs):
            feature = _Diffusion()
            features.append(feature)
            return feature

        def fake_test_single(*positional, **kwargs):
            time_index = positional[4]
            calls = validation_calls.get(time_index, 0)
            validation_calls[time_index] = calls + 1
            score = (0.9 if time_index == 1 else 0.8) if calls == 0 else 0.1
            return (torch.tensor(score),
                    [torch.tensor(0.5)] * 3, torch.tensor([1]))

        def fake_final_test(*positional, **kwargs):
            final_evaluations.append((positional, kwargs))
            if getattr(args, "fahc_final_validation", False):
                return (0.1, 0.4, 0.1, 0.2, "changed history", "changed output",
                        [0.3, 0.5], [], [], [], [0.3], [], [], [])
            return (0,) * 14

        validation_calls = {}

        def fake_probe(model, teacher, candidates, current, history,
                       static_graph, use_cuda, task_weight, learning_rate,
                       grad_norm, steps, batch_size, *, training_optimizer):
            parameter = next(model.parameters())
            moment = float(training_optimizer.state[parameter]["exp_avg"])
            probe_observations.append((
                [len(candidate.triples) for candidate in candidates],
                len(current), moment))
            return {candidate.time_index: 1.0 for candidate in candidates}

        def spy_select(candidates, *positional, **kwargs):
            selection_observations.append(
                [len(candidate.triples) for candidate in candidates])
            return fahc_selector.select_hcp(candidates, *positional, **kwargs)

        def spy_materialize(selection, full_candidates, budget, seed):
            materialize_observations.append(
                (selection.fact_count, budget,
                 [len(candidate.triples) for candidate in full_candidates]))
            return fahc_selector.materialize_replay(
                selection, full_candidates, budget, seed)

        data = SimpleNamespace(train=self.snapshots[0], valid=self.snapshots[1],
                               test=self.snapshots[2], num_nodes=12, num_rels=2)
        with ExitStack() as stack:
            for target, replacement in (
                ("load_data", lambda dataset: data),
                ("load_all_answers_for_time_filter",
                 lambda snapshots, *rest: [{} for _ in snapshots]),
                ("get_total_rank", lambda *a, **kw:
                 (torch.tensor(0.5), torch.tensor(0.5),
                  torch.tensor([1]), torch.tensor([1]))),
                ("stat_ranks_record", lambda *a, **kw:
                 (torch.tensor(0.5), [torch.tensor(0.5)] * 3)),
            ):
                stack.enter_context(patch.object(main.utils, target, replacement))
            for target in ("add_hyper_in_file", "add_other", "finish"):
                stack.enter_context(patch.object(main.fitlog, target,
                                                 lambda *a, **kw: None))
            stack.enter_context(patch.object(main, "validate_args",
                                             lambda args: None))
            stack.enter_context(patch.object(main, "experiment_signature",
                                             lambda args: "testsig"))
            stack.enter_context(patch.object(main, "build_sub_graph",
                                             lambda *a, **kw: object()))
            stack.enter_context(patch.object(main, "create_model_diffu",
                                             lambda args: None))
            stack.enter_context(patch.object(main, "Att_Diffuse_model",
                                             fake_feature))
            stack.enter_context(patch.object(main, "RecurrentRGCN",
                                             lambda *a, **kw: _Model()))
            stack.enter_context(patch.object(main, "EarlyStopping",
                                             FakeEarlyStopping))
            stack.enter_context(patch.object(main, "test_single",
                                             fake_test_single))
            stack.enter_context(patch.object(main, "test",
                                             fake_final_test))
            stack.enter_context(patch.object(main, "tqdm", lambda items: items))
            stack.enter_context(patch.object(main.torch, "load", fake_load))
            stack.enter_context(patch.object(torch.optim.Adam, "step",
                                             fake_step))
            stack.enter_context(patch.object(main, "probe_hcp_risks",
                                             fake_probe))
            stack.enter_context(patch.object(main, "select_hcp", spy_select))
            stack.enter_context(patch.object(main, "materialize_replay",
                                             spy_materialize))
            stack.enter_context(patch.object(main, "load_hcp_candidates",
                lambda dataset, indices, subjects:
                    fahc_selector.load_hcp_candidates(
                        dataset, indices, subjects, self.history_dir)))
            with redirect_stdout(io.StringIO()):
                result = main.run_experiment(args)

        records = []
        if self.log_path.exists():
            records = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        return SimpleNamespace(
            result=result, checkpoints=checkpoint_store, load_paths=load_paths,
            features=features, probes=probe_observations,
            selections=selection_observations,
            materializations=materialize_observations, records=records,
            final_evaluations=final_evaluations)

    def test_final_validation_uses_validation_labels_and_same_training_context(self):
        args = self._args(mode="original", diagnostics=True)
        args.fahc_final_validation = True
        run = self._run(args)
        self.assertEqual([kwargs["mode"] for _, kwargs in run.final_evaluations],
                         ["test", "valid"])
        first, second = run.final_evaluations
        self.assertIs(first[1]["last_history"], second[1]["last_history"])
        self.assertIs(first[1]["last_output"], second[1]["last_output"])
        self.assertTrue(all(len(snapshot) == 1 for snapshot in second[0][3]))
        self.assertTrue(all(snapshot[0, 2] == 9 for snapshot in second[0][3]))
        record = next(record for record in run.records
                      if record["event"] == "final_validation")
        self.assertEqual(record["past_macro_filtered_entity_mrr"], 0.3)
        self.assertEqual(record["current_filtered_entity_mrr"], 0.5)
        self.assertEqual(record["filtered_entity_mrr"], 0.4)

    def test_mechanism_prefix_stops_and_never_evaluates_test(self):
        args = self._args(mode="random")
        args.fahc_mechanism_stages = (3,)
        snapshots = [_snapshot(index) for index in range(5)]
        splits = ([value[:8, :3] for value in snapshots],
                  [value[8:9, :3] for value in snapshots],
                  [value[9:, :3] for value in snapshots])
        metric = dict(spearman_ab=0.5, spearman_mean_observed=0.4,
                      observed_topk_lift=0.1)
        record = dict(event="mechanism", snapshot=3,
                      metrics={"current_only": metric, "dgar_history_only": metric})
        with (patch.object(main.utils, "spilt_data", return_value=splits),
              patch.object(main, "begin_mechanism_diagnostic", return_value={}) as begin,
              patch.object(main, "finish_mechanism_diagnostic", return_value=record) as end):
            run = self._run(args)
        self.assertEqual(begin.call_count, 1)
        self.assertEqual(end.call_count, 1)
        self.assertEqual(run.result, (None,) * 4)
        self.assertEqual(run.final_evaluations, [])
        completed = [value for value in run.records if value["event"] == "mechanism_complete"]
        self.assertEqual(completed[0]["snapshots"], [3])
        stages = [value["snapshot"] for value in run.records if value["event"] == "stage"]
        self.assertEqual(stages, [1, 2, 3])

    def test_t1_warmup_then_t2_full_hcp_selection_precedes_fact_cap(self):
        run = self._run(self._args())
        self.assertEqual(run.probes, [([8], 2, 1.0)])
        self.assertEqual(run.selections, [[8]])
        self.assertEqual(run.materializations, [(8, 2, [8])])
        selected_calls = [facts for time, facts in run.features[1].feature_calls
                          if time == 2 and facts is not None]
        self.assertEqual(len(selected_calls), 1)
        self.assertEqual(len(selected_calls[0]), 2)
        stage_two = next(record for record in run.records
                         if record.get("event") == "stage" and record["snapshot"] == 2)
        self.assertEqual(stage_two["candidate_fact_counts"], {"0": 8})
        self.assertEqual(stage_two["scored_fact_counts"], {"0": 8})
        self.assertEqual(stage_two["selected_fact_count"], 2)

    def test_original_diagnostics_without_diffusion_skips_diffusion_checkpoint(self):
        args = self._args(mode="original", diffusion=False, diagnostics=True)
        args.delete_feature_reply = True
        run = self._run(args)
        self.assertFalse(run.probes)
        self.assertFalse(run.selections)
        self.assertFalse(any("diffu_models" in path for path in run.load_paths))
        self.assertFalse(any("diffu_models" in path for path in run.checkpoints))
        self.assertEqual([record["event"] for record in run.records],
                         ["run_config", "stage", "stage"])

    def test_early_stop_before_start_reply_still_saves_diffusion_checkpoint(self):
        args = self._args(mode="original", diffusion="add", start_reply=1)
        run = self._run(args, stop_after_first_epoch=True)
        self.assertTrue(any("diffu_models" in path for path in run.checkpoints))
        self.assertTrue(any("diffu_models" in path for path in run.load_paths))


if __name__ == "__main__":
    unittest.main()
