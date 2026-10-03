# DGAR

Run the original model from `src`:

```bash
cd src
python main.py --gpu 0 --dataset ICEWS14s
```

## FAHC-DGAR

FAHC selects historical context prompts (HCPs) using a short current-data
probe. At each stage it loads the previous best model, scores HCP facts before
and after a probe update, then restores all model parameters, BatchNorm buffers,
and random-number state. A separate probe optimizer leaves DGAR's Adam state
unchanged. Training then uses the selected HCP facts in the existing diffusion
feature/replay path. Optional old-model entity distillation is gated by teacher
correctness and confidence. Its KL uses history-only entity logits for both
teacher and student; DGAR's main replay loss remains diffusion-conditioned.

The risk of a snapshot is the mean positive per-fact cross-entropy increase.
The probe uses at most 128 current facts by default (`--fahc-probe-facts`).
Risk scoring uses the same current history graph as DGAR replay, with the
diffusion feature omitted from the probe. Coverage uses diminishing harmonic
gains for relation IDs, entity IDs, and time bins. Set
`--fahc-entity-weight 0` for the relation/time-only objective.
Inverse relation IDs share their base relation's coverage count.
Under a fixed fact budget, each candidate first receives a seeded fact quota
for scoring. Every scored fact is retained if its HCP is selected; remaining
capacity is filled from the selected slices. This avoids choosing a slice for
facts that are then discarded by the replay cap.

Run controlled HCP comparisons with the **same** `--reply_batch` (number of
historical slices) and `--fahc-fact-budget` (maximum unique replay facts):

```bash
cd src
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection random --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection recent --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection coverage --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk+coverage --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk+coverage --reply_batch 35 --fahc-fact-budget 128 --fahc-kd-weight 0.1
```

`--hcp-selection original` (the default) retains the original DGAR sampler
and its uncapped normal HCP path. For an HCP-KD ablation with a matched fact
budget, use `--hcp-selection random --fahc-kd-weight 0.1`. All controlled
modes draw from DGAR's original eligible snapshots, `0..t-2`; `recent` picks
the newest `k` among them. The candidate multiplier defaults to 2, and can
be set to 2–4 with `--fahc-candidate-multiplier`. The actual replay fact count
can fall below the cap if selected slices contain too few relevant facts; the
JSONL log records it for fair analysis.

FAHC writes per-stage candidate/selected indices, risks, replay fact counts,
KD loss, and current/past history-only filtered MRR to `logs/fahc_*.jsonl`.
This diagnostic evaluates four evenly spaced old validation snapshots per
stage by default; use `--fahc-past-eval-samples 0` for every old snapshot, or
`--fahc-eval-every N` to change old-snapshot frequency. Current-snapshot MRR
is recorded every stage when evaluation is enabled. It is separate from DGAR's
diffusion-conditioned validation metric. FAHC checkpoints get a configuration
hash in their names so runs do not overwrite the original DGAR checkpoints.
For a comparable full forgetting curve across stages, set
`--fahc-past-eval-samples 0`; the default sampled macro uses changing old
snapshot cohorts and is intended as a quick diagnostic.

This version reuses all existing `data/*/history_snap_v2` files. It limits
replayed facts per stage, **not total historical storage**. A fixed-byte online
HCP cache would require a separate experiment before claiming bounded memory.
With KD enabled, cached teacher logits use memory proportional to the replay
fact budget times the entity count.

From the repository root, run the focused checks with an environment
containing PyTorch, DGL, and NumPy:

```bash
python -m unittest discover -s tests -p 'test_fahc*.py'
```
