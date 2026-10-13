# GRB-DGAR: Gradient-balanced Replay

## Choice and evidence

Independent branch `codex/grb`, from clean `codex/original-baseline` at
`23dd43dd914a783731c7f39e3fd66491d2aba429`. The scientific baseline restoration
is `bedb5b1`; `23dd43d` only adds its completed ICEWS14s reproduction report.
Author source and protocol remain pinned by [BASELINE.md](BASELINE.md).

Previous QCDR experiments provide a direction to test, not a guaranteed gain:
candidate stop-gradient lost entity MRR; candidate alpha=.5 did not beat the
simple control that halves BOTH entity replay CEs. Those experiments predate
the official baseline restoration and cannot be subtracted from this baseline.
They also do not prove relation-specific interference or temporal regime changes.

The lowest-risk next experiment therefore balances the total entity replay
contribution. GRB retains Original's full contribution when measured replay
is compatible and not dominant, and decreases it when it is dominant or opposed
to current supervision. The default floor .5 preserves substantial historical
learning. This is a conservative optimization hypothesis, not a demonstrated
paper contribution. It must outperform BOTH Original and fixed half to justify
the adaptive mechanism. If fixed half wins, prefer the simpler control.

Unlike a relation-specific design, the first version does not infer unsupported
relation evolution from sparse batches. A relation-aware extension would require
separate evidence. Unlike QCDR, it scales both query and candidate replay gradients.
There is no new teacher, KD, HCP selection, fact budget, probe update or fusion gate.
Original's existing feature-generator teachers and scalar fusion alpha remain.

## Definition

In one predictor update, use Original's existing CE terms:

\[
L_c=\mathrm{CE}(D_t^{\pm}),\quad
L_f=\mathrm{CE}(\text{replay forward}),\quad
L_i=\mu\,\mathrm{CE}(\text{replay forward+inverse}),\quad L_p=L_f+L_i.
\]

Let H be the final normalized entity representation immediately before decoding
(`pre_emb`, shape N by d). From the SAME forward computation:

\[
g_c=\partial L_c/\partial H,\qquad g_p=\partial L_p/\partial H.
\]

The measured gradients include both direct query and candidate paths through H.
They are not full parameter gradients: indirect relation-state routes, relation
CE, static loss and decoder parameter gradients are not measured. A negative
cosine at H does not prove catastrophic forgetting or harmful Adam updates.

Within a stage, EMA the three scalar moments, initialized by the first valid batch:

\[
(A,B,C)\leftarrow m(A,B,C)+(1-m)(\|g_c\|^2,\|g_p\|^2,\langle g_c,g_p\rangle).
\]

\[
c=\operatorname{clip}\!\left(\frac{C}{\sqrt{AB}},-1,1\right),\qquad
w=\operatorname{clip}\!\left(
\frac{\rho\sqrt A}{\sqrt B\,(1+\gamma[-c]_+)}, w_{\min},1\right).
\]

Then use a DETACHED weight:

\[
L=\tau(L_c+wL_f+wL_i)+(1-\tau)L_r+L_{\rm static}.
\]

Defaults: rho=1, gamma=1, w_min=.5, EMA=.8; tau is Original's task_weight=.7.
Zero/unavailable/nonfinite gradient information falls back to weight 1; invalid
observations do not poison EMA. The controller does not repair nonfinite losses.
The floor is a retention heuristic, not a guarantee that current loss will decrease.

`autograd.grad(..., H, retain_graph=True, create_graph=False)` measures the two
terms without accumulating parameter `.grad`. It adds two decoder backward
traversals, not a second RGCN forward/backward. The normal total-loss backward,
gradient clipping and Adam update are performed once. No gradient projection,
second-order derivatives, random draws or optimization of the weight are used.

## Causal data and call flow

1. Main validates config, uses Original initialization/seed policy, data split and
   immutable history cache, builds the existing models, optimizer and graphs.
2. At chronological stage t, reset the GRB scalar EMA and stage records. Training
   HCP and generated representations follow exactly Original's past-index rules.
3. get_loss performs the same graph forward and decoder calls in the same order,
   retains references to the existing current and two entity replay CE scalars.
4. Only in TRAIN with grad enabled, pass H and those scalars to ReplayBalancer.
   Compute a weight and rebuild the entity loss; relation/static terms are untouched.
5. Main executes original backward/clip/Adam, validation, weights-only best reload
   with last-update Adam, diffusion training and teacher synchronization.
6. After the existing stage checkpoint, write grb-stages.jsonl and a one-line log.
7. TEST/VALIDATION/diffusion guidance do not call the controller. Original's pinned
   retrospective evaluation remains; it is not converted to strict online testing.

Only stage training labels and already generated replay enter the controller.
No held-out labels, future training snapshots, accuracy-based teacher filter or
future statistics determine w. Do not describe the inherited retrospective final
evaluation as strictly causal just because the new controller uses causal inputs.

No model keys, replay memory, prototypes, persistent teacher or auxiliary
checkpoint state are added. EMA and records detach their tensors. EMA resets at
every stage; latest.pt resumes only after completed stages, so restored model,
Adam and RNG plus a fresh EMA reproduce the next stage. Intra-stage resume is
not supported by Original or this method. Different configs/source are rejected.
Use new run directories rather than old Original/FAHC/ECRI/QCDR checkpoints.

The measurements and controller stay on H's device. Two N*d FP32 gradients and
temporary reductions add O(Nd) storage/work. One such ICEWS14s tensor is about
5.44 MiB; this is not a measured peak-memory estimate. The actual CUDA time and
memory overhead have not been benchmarked. Only scalar records are copied to
CPU once per stage; the baseline's pre-existing loss.item() calls are retained.

## Modes and parameters

| Argument | Default | Meaning |
|---|---|---|
| --grb-mode | original | original (OFF), fixed (control), adaptive (GRB) |
| --grb-fixed-weight | .5 | Fixed mode scales both entity replay CEs |
| --grb-min-weight | .5 | Adaptive weight floor |
| --grb-target-ratio | 1 | Desired replay/current representation-gradient norm ratio |
| --grb-conflict-scale | 1 | Additional damping for negative cosine; 0 disables it |
| --grb-ema | .8 | Within-stage smoothing; 0 uses the current update only |

All scientific baseline defaults are preserved. OFF initializes no controller,
does no measurements/log writing, and preserves loss addition and decoder order.
Fixed weight 1 bypasses loss rebuilding exactly. Feature teacher TRAIN modes,
shared RNG, HCP set order, best-weight/last-Adam semantics and final history flow
are covered by inherited protocol tests. No changes to decoder or diffusion files.

Stage JSONL stores current/replay/weighted-replay CE, gradient norms, cosine,
valid-signal flag, each update's weight and stage aggregates. Original has no GRB
file. Fixed mode records losses/weights but deliberately does not measure gradients;
its norm/cosine placeholders have valid_signal=0 and must not be interpreted.
Summary tooling deduplicates stages by the last record if a resume appends one.

## Files and checks

Modified: src/cli.py (six method/control options), src/main.py (controller lifecycle,
call and stage logging), src/rrgcn.py (retain CE components, train-only combine).
Added: src/grb.py, tests/test_grb.py, scripts/run_grb_experiment.sh,
scripts/summarize_grb.py, this report. README links the method and commands.

The required CPU suite checks pinned Original training/evaluation/HCP behavior;
real RGCN+decoder OFF/fixed1 vs an independently loaded baseline loss, gradients,
buffers, optimizer step and RNG exactly; manual vs controller measured gradients;
affine fixed-weight gradients; singleton/inverse/empty replay; detached EMA and
records; no evaluation intervention; stage-boundary resume; an actual main caller
with a synthetic replay backward stopped before any optimizer step or epoch.
Also run compilation, --help, shell syntax/dry-run and summary parser checks.

No real data training or GPU experiments are executed during implementation.
Numerical equality on CPU is not a performance claim. CUDA/DGL behavior, overhead,
actual gradient-signal stability and generalization improvement require the three
matched experiments below. No improvement probability is numerically established.

Implementation validation (2026-10-13): all 25 CPU tests passed in 1.526 seconds
(excluding interpreter import); 27 Python files compiled, actual main --help
passed. bash -n and adaptive dry-run passed. The real training-caller test stops
after a synthetic backward and forbids optimizer.step; no real epoch is entered.
One initial check used the base Python without NumPy; all reported checks use
the existing dgar_env interpreter. No dependency installation was required.

## First experiment

Use one explicit seed (1), one physical GPU (0), the same environment/data/cache,
and run the three groups SEQUENTIALLY. This screens a direction without extra seeds.
The existing unseeded Original reproduction is a source/protocol reference, not
the matched control for this experiment. New Original control is necessary.

```bash
cd /data1/zihao/DGAR-code/.worktrees/grb
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"  # after activating dgar_env
DGAR_PYTHON="$DGAR_PYTHON" bash scripts/check_baseline.sh
bash scripts/run_grb_experiment.sh --dry-run original 0 ICEWS14s 1
bash scripts/run_grb_experiment.sh --dry-run half 0 ICEWS14s 1
bash scripts/run_grb_experiment.sh --dry-run adaptive 0 ICEWS14s 1
```

Run ONE line, wait for completion, then the next. The sequential all-three version
is convenient if GPU0 can be reserved for roughly three baseline-length runs:

```bash
GRB_LAUNCH_LOG="/data1/zihao/DGAR-code/.worktrees/grb/logs/grb-three-groups-$(date +%Y%m%d_%H%M%S)_$$.log"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/grb
bash scripts/run_grb_experiment.sh original 0 ICEWS14s 1
bash scripts/run_grb_experiment.sh half 0 ICEWS14s 1
bash scripts/run_grb_experiment.sh adaptive 0 ICEWS14s 1
' < /dev/null > "$GRB_LAUNCH_LOG" 2>&1 &
printf 'Launch log: %s\n' "$GRB_LAUNCH_LOG"
```

Create logs first (`mkdir -p logs`) for that outer redirect; each script generates
a unique run name and its own logs/<run-name>.log with tee. Python is unbuffered,
pipefail propagates failures, and no failed group proceeds to the next. All groups
use Original regcn_epochs/diffu_epochs=10, mu=mu_r=1, reply_batch=35, no budget/KD.

```bash
ls -lt /data1/zihao/DGAR-code/.worktrees/grb/logs
tail -f "$GRB_LAUNCH_LOG"
```

After completion, replace the three directory placeholders with actual names:

```bash
python scripts/summarize_grb.py \
  --original experiment_runs/ORIGINAL_RUN_NAME \
  --half experiment_runs/HALF_RUN_NAME \
  --adaptive experiment_runs/ADAPTIVE_RUN_NAME \
  --output experiment_runs/grb-summary-first-three
```

The parser requires matching source/data/cache/protocol/scientific settings and
GPU index; use one server/environment/physical GPU anyway. It cannot prove that
GPU index 0 is the same device on two different machines. It reads only configs,
logs and JSONL; no inference/training/torch import. Metrics.csv and summary.json
preserve TEST/VALIDATION separately, macro vs pooled, H1/3/10 and weight distribution.

Decision: adaptive must improve final VALIDATION pooled AND historical macro MRR
over BOTH controls, with no meaningful H1 loss, and show useful nonconstant weights.
TEST is a separate report, not a tuning target. If adaptive equals fixed half,
or its weights nearly always hit .5, there is no extra adaptive contribution.
If half wins, stop this controller rather than sweep fine-grained rho/gamma/EMA.
If Original wins, stop claiming entity replay should be suppressed on this baseline.
One seed provides screening evidence, not a statistically established paper claim.

## Transfer to another server

After this branch is pushed, on a server without this worktree:

```bash
git -C /data1/zihao/DGAR-code fetch origin
git -C /data1/zihao/DGAR-code worktree add --track -b codex/grb \
  /data1/zihao/DGAR-code/.worktrees/grb origin/codex/grb
```

For an existing clean GRB worktree, use `git pull --ff-only` there instead.
Activate that server's dgar_env, retain its matching Original data/history cache,
then use the commands above. Do not push the old local HEAD to this new branch.
