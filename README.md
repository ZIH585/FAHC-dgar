# DGAR / QCDR

Independent `codex/qcdr` branch: Original DGAR with shared runtime fixes, plus
optional **Query–Candidate Decoupled Replay**. QCDR changes only the direct
candidate gradient in historical entity replay CE. No new encoder, selector,
teacher, replay budget, or auxiliary loss.

- [Baseline provenance and shared fixes](docs/BASELINE.md)
- [Implementation, data flow, gradient contracts and verification](docs/QCDR.md)
- [Five complete experiment commands, logs and resume](docs/QCDR_EXPERIMENTS.md)
- [Entity replay weight control and three comparison commands](docs/QCDR_REPLAY_CONTROL.md)

The default `--qcdr-mode original` runs the corrected Original control.
`candidate-stopgrad` cuts the direct candidate branch only during entity replay;
`candidate-scaled --qcdr-candidate-alpha 0.25` retains 25% of that gradient.
Current learning and relation prediction retain their original gradient paths.
The control-only `--qcdr-replay-entity-weight` defaults to 1; setting it to 0.5
scales BOTH historical entity CE terms, including query and candidate gradients.

Datasets and the matching `history_snap_v2` cache are supplied separately via
`--data-root`; they are not tracked by this branch. Use a new `--run-dir` per run.

CPU checks (activate the existing DGAR environment; no real training):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -v
```

The tests use synthetic tensors/graphs and compare against baseline `a4f00eb`
from Git history. Keep that ancestor available when cloning.
