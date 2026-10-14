# RTER implementation and experiments

## Baseline and scope

Branch `codex/rter` starts exactly at `codex/original-baseline@23dd43dd914a783731c7f39e3fd66491d2aba429`.
That is the restored official-DGAR behavior, not a FAHC/ECRI/QCDR/BCR/DCR
method branch. The author reference is `zyzhang11/DGAR@c5bbc87467bb22d266b664d30eefb8184c1d3104`.
All files under diffusion/ and rgcn/, src/rrgcn.py, src/decoder.py are unchanged.
The generator's original relation/entity indexing and sampling endpoint are
preserved. RTER is a scoring method, not a new replay selector or optimizer.

`--rter-mode original` (default) skips the evidence index entirely. Positive
enabled beta uses protocol `rter-score-v1`; OFF stays `official-dgar-v1`.
The score operator with beta 0 is identity. Changing the mode/beta in an existing
main.py --run-dir is refused by provenance checks: reuse an Original checkpoint
with scripts/evaluate_rter.py instead of modifying its config/checkpoint.

## Method and causal order

For q=(s,r,?,t), event evidence is

    z_t(s,r',e) = sum_(tau<t) exp(-(t-tau)/h) 1[(s,r',e) in train_tau]

An ordered entity pair has an observed relation set per snapshot. Only transitions
between consecutive *observed* snapshots are counted, with both timestamps <t
and gap<=window. Equal-time relations never transition to each other. A Cartesian
transition between sets R0,R1 receives 1/(|R0| |R1|) per pair, so its total mass
is 1. Repeated rows and supplied inverse duplicates do not inflate evidence.
Missing events are not negative labels. Transition mass decays by the same h.

To prevent inverse augmentation creating a trivial direction-only lift, target
forward and inverse relation blocks have separate marginals and normalization:

    p_t(r | target_direction) = target marginal in that relation block
    P_t(r | r', target_direction)
       = (N_t(r',r) + a*p_t(r)) / (row_mass_in_target_block + a)
    A_t(r',r) = max(log(P_t/p_t),0)
    b_t(q,e) = min(log(1 + sum_(r') A_t(r',r)*z_t(s,r',e)), clip)
    scores_new = scores_DGAR + beta*b_t

Default cross excludes r'=r; all includes identity transitions. Both directions
of a historical observed training triple are legitimate evidence. For an inverse
query, its known subject is the original object, as in the author protocol. The
evidence API accepts exactly [query_subject,query_relation], never target labels.
Inverse queries are not used to share their hidden targets as retrieval anchors.
All entities remain candidates. Unsupported candidates get residual zero.

## Data flow and training/evaluation calls

1. Original load_data/split_by_time/spilt_data organizes train/valid/test. In
   particular, raw train.txt alone is not the effective entire training stream.
2. Original chronological predictor/diffusion optimization, replay construction,
   guidance, get_loss and per-stage validation/early stopping run unchanged.
3. Final test(): a fresh RelationEvidence holds references to training arrays;
   advance(t) reads only indices 0..t-1, never the current/future slice.
4. Original HCP generation + full forward/inverse decoder batch produces scores.
5. RTER adds the sparse bounded candidate residual before the original filtered
   ranking. Relation scores are unchanged.
6. Final supplementary VALIDATION rebuilds its own prefix index from time zero.

No old model, KD, probe, new neural parameters or additional training loss.
Evidence is reconstructed, not serialized in model.pt/latest.pt. Resume training
does not need an evidence state: it is only used after training. Default OFF
never allocates this index and preserves training RNG/buffers/Adam rollback.

The parent evaluation is retrospective: the model has trained through the final
snapshot, and its first generator-guidance history follows the author quirk.
RTER statistics are query-time causal; this does not turn the full DGAR protocol
into strictly online forecasting. No future-label-trained evidence estimator is
introduced. Training facts at t are excluded from RTER when evaluating t.

## Parameters

|Flag|Default|Meaning|
|---|---|---|
|--rter|not supplied|Alias for --rter-mode transition|
|--rter-mode|original|OFF / transition / repeat / popularity / shuffled|
|--rter-beta|0.1|Final residual coefficient, nonnegative|
|--rter-horizon|30|Snapshot e-folding time h (not literal half-life)|
|--rter-smoothing|20|Transition prior pseudocount a|
|--rter-window|30|Maximum observed-event transition gap|
|--rter-clip|3|Bound on unit log-evidence before multiplying beta|
|--rter-channel|cross|Exclude identity relations; all is an ablation|
|--rter-shuffle-seed|1|Local deterministic forward/inverse-consistent permutation|

repeat uses exact (s,r,e) historical counts, popularity uses global historical
endpoint frequency, shuffled permutes both axes of learned affinities and retains
target direction blocks. All use the same time cutoff, decay, clipping and score
operator. Independent beta selection for each control on development VALIDATION
avoids giving RTER a tuning advantage. There is no claim of bounded total memory:
the sparse history index grows with distinct historical entity/relation pairs.
For ICEWS14s the float64 transition-count matrix is about 1.61 MiB on CPU;
affinities/temporary arrays and Python history dictionaries add memory. No extra
model copy is introduced during main training. The paired evaluator loads the
existing predictor/generator and adds one score tensor transiently per arm.

## Recommended first experiment: reuse Original, no retraining

Activate dgar_env first. The following existing run is audited Original, despite
its directory being in the DCR worktree (dcr_mode=original). It used GPU index 1.
Run on an idle GPU; keep the same GPU/environment for the comparison. The script
checks source/data/cache, completed stage, finite weights and best/latest equality.
Supported sources are the restored baseline and audited GRB/DCR OFF commits,
plus this branch's own OFF source fingerprint. Enabled method weights are refused.

```bash
cd /data1/zihao/DGAR-code/.worktrees/rter
export DGAR_PYTHON="$(command -v python)"
mkdir -p logs

# Optional: print command only; no GPU/data inference, no outputs created.
bash scripts/run_rter_evaluation.sh --dry-run validation 1 \
  /data1/zihao/DGAR-code/.worktrees/dcr/experiment_runs/ICEWS14s-dcr-original-seed1-20261014_143019_4046517

# Launch only when ready. No optimizer or training epoch is executed.
setsid bash -c '
set -o pipefail
bash scripts/run_rter_evaluation.sh validation 1 \
  /data1/zihao/DGAR-code/.worktrees/dcr/experiment_runs/ICEWS14s-dcr-original-seed1-20261014_143019_4046517 \
  2>&1 | tee logs/ICEWS14s-rter-validation-launcher-$(date +%Y%m%d_%H%M%S)-$$.log
' < /dev/null > /dev/null 2>&1 &
```

Each invocation prints and uses fresh experiment_runs/rter-validation-* and
logs/rter-validation-*.log. The outer launcher also records these paths. Inspect
the selected launcher with tail -f. DGAR_DATA_ROOT can override the shared data
root; changing physical paths does not bypass provenance verification.

The evaluator first performs the author full TEST trajectory to recreate the
supplementary VALIDATION RNG/buffers. It scores RTER variants on VALIDATION only
in this first phase; TEST warmup runs Original and is not used for beta selection.
Unlike the earlier sampled diagnostics, there is no per-stage RNG isolation or
fresh reseeding. Each stage gets one original generation/prediction; controls
share those exact logits. Original score filtering is in-place, so every control
passes a protected clone to the rank function.

Default beta grid is 0.1/0.3/1; no horizon/smoothing sweep. Full VALIDATION is
split before examining results: development 1..182, confirmation 183..363 for
ICEWS14s, with last stage 364 reported separately. Select beta by development
macro MRR, tie toward smaller beta. Existing repeatedly inspected validation
results remain exploratory; this split is a resource decision, not independence
from all earlier research choices.

Outputs:

- summary.json: per-arm pooled MRR/H1/H3/H10, macro MRR, per-stage, direction,
  repeated/unseen groups, independently selected control betas, confirmation
  metrics, win/loss/tie counts and top-5 gain concentration.
- queries.csv: paired ranks, time/direction and past-fact status. Labels appear
  here only after prediction, for offline reporting.
- rter.jsonl: history_stop, decayed transition mass, index size, supported queries,
  boosted entries and residual magnitude. No label-dependent evidence statistics.
- training/analysis provenance and original full TEST/VALIDATION metrics in log.

The automatic provisional gate requires confirmation macro and pooled MRR each
>= Original+0.3pp, macro MRR greater than the independently tuned repeat,
popularity and shuffled controls, and neither direction pooled MRR <-0.3pp.
Additionally inspect unseen facts and concentrated gains. This is not a p-value
or proof of generalization. Failure means stop this configuration before a large
training sweep.

## Locked TEST after a successful confirmation

Use the exact summary.json path printed by the first experiment. A passed
selection-file locks the beta and all evidence settings, and verifies the same
model/data/cache/RTER source. TEST does not permit a multi-beta grid.

```bash
# Replace this one path with the completed validation output directory.
export RTER_SELECTION=/data1/zihao/DGAR-code/.worktrees/rter/experiment_runs/rter-validation-RUN_ID/summary.json
setsid bash -c '
set -o pipefail
bash scripts/run_rter_evaluation.sh test 1 \
  /data1/zihao/DGAR-code/.worktrees/dcr/experiment_runs/ICEWS14s-dcr-original-seed1-20261014_143019_4046517 \
  "$RTER_SELECTION" \
  2>&1 | tee logs/ICEWS14s-rter-test-launcher-$(date +%Y%m%d_%H%M%S)-$$.log
' < /dev/null > /dev/null 2>&1 &
```

Original and RTER TEST are paired in this single run. There is no need to train
four new neural models or retrain the existing Original to evaluate RTER.
If TEST gains survive and benefit crosses queries/relations, next test the locked
method on another dataset's own Original checkpoint. Combine DCR mean only after
RTER has its independent evidence; the initial evaluator deliberately refuses
DCR-enabled weights.

## Optional full training interface

This is available for future fresh dataset reproductions, not required now:

```bash
bash scripts/run_rter_experiment.sh --dry-run original 1 ICEWS14s 1
bash scripts/run_rter_experiment.sh --dry-run transition 1 ICEWS14s 1 0.1
# Remove --dry-run to train when explicitly ready.
```

It uses unique run directories, --final-validation, seed1 and the normal DGAR
replay defaults. These two neural training trajectories should be identical in
the same environment because RTER acts after training. No claim of training
forgetting improvement can be inferred from a score-only gain.

## Checks executed during implementation

scripts/check_baseline.sh disables CUDA and runs the inherited author regression
plus RTER CPU tests. Tests cover prefix causality, empty/first stage, singleton,
inverse/deduplication, target-free API, dtype/shape, decay/window/time ties,
direction-marginal controls, local RNG, bounded residuals, rebuilding state,
unchanged enabled training loop, actual RGCN final-score/log integration,
development-only beta selection and a tiny real saved-checkpoint evaluator.
The toy script compares different control counts to catch shared-logit mutation,
and verifies byte-for-byte unchanged inputs. --help, imports, py_compile and bash
syntax checks are included. The final suite passed 28 tests in about 1.8 seconds
and compiled 29 Python files. Fifteen model/diffusion/graph source files matched
the baseline byte-for-byte. A real ICEWS14s Original CPU --check-only also
validated 365 stages/7,128 entities/230 relations, stopping before inference.

No real training, GPU inference or full real-data evaluation was run by the agent.
Actual performance, peak memory, wall-clock cost and CUDA/DGL behavior need the
user's experiment. Necessary new computation is CPU historical indexing and
candidate score updates; no constant-time/memory claim is made.

## Git and another server

```bash
git -C /data1/zihao/DGAR-code fetch origin codex/rter
git -C /data1/zihao/DGAR-code worktree add \
  -b codex/rter /data1/zihao/DGAR-code/.worktrees/rter origin/codex/rter
```

If that branch/worktree already exists, use its clean worktree and
git pull --ff-only origin codex/rter. Do not switch/reset the dirty root checkout.
Datasets/cache/checkpoints/experiment outputs are ignored and not pushed. Copy
the required audited Original run if the other server does not already have it;
preserve config.json, latest.pt, model.pt, diffusion.pt together.
