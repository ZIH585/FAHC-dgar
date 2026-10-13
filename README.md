# DGAR + Gradient-balanced Replay (GRB)

Branch: `codex/grb`, from `codex/original-baseline@23dd43d`.
Protocol: `official-dgar-v1`. Default method mode is OFF (`--grb-mode original`).
Pinned author source: `zyzhang11/DGAR@c5bbc87467bb22d266b664d30eefb8184c1d3104`.

Read [the baseline audit](docs/BASELINE.md) before running experiments or branching
a new method. See [GRB definition, checks and three-group experiments](docs/GRB.md).
No previous FAHC/ECRI/QCDR implementation or KD/selector flags are included.

```bash
# CPU checks only; no dataset or GPU training.
DGAR_PYTHON=/path/to/dgar_env/bin/python bash scripts/check_baseline.sh

# Training example: execute only when ready; choose a fresh output directory.
python -u src/main.py --gpu 0 --dataset ICEWS14s \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir experiment_runs/ICEWS14s-official-v1
```

Omitting `--seed` preserves the author's initialization policy. An explicit
`--seed 1` opts into reproducible Python/NumPy/Torch initialization; both sides
of a method comparison must use the same policy. Validation consumes the same
random stream as training, matching the author. Preserve the original HCP cache.

Final TEST is the author's retrospective evaluation protocol. Read the audit's
history-flow explanation before interpreting it as online forecasting. The
paper's Average MRR and pooled filtered MRR are different aggregates.
