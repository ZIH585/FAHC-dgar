# Original DGAR provenance

The source starts directly at `a454564` (`master` / `origin/master`), before
`76c2c35` introduced the prior experimental selector. No later research commit
is an ancestor of this branch. Algorithm-specific selection, probing, KD,
budgets, experiment launchers and diagnostics were not copied.

The following common fixes are applied equally to Original and QCDR:

- CPU/device-safe tensors and graph metadata; string comparison; empty replay;
  singleton relation decoder BatchNorm; finite static-loss denominator.
- Typed booleans/numeric CLI arguments; working-directory-independent data paths;
  reproducible Python/NumPy/PyTorch seeds and isolated validation randomness.
- Restore each stage's best model **and matching Adam state** before the next
  stage, and keep the existing frozen feature teachers synchronized.
- Diffusion is trained at every enabled stage. Unsupported nonzero start_reply
  is rejected rather than leaving a stale/missing diffusion checkpoint.
- Final evaluation guidance uses the queried snapshot's strictly past training
  graph. Original accidentally initialized early-query guidance with the last
  training stage's history and later used lagged evaluation history.
- Remove the redundant last-stage TEST peek during training. Final TEST retains
  the original query set, inverse directions, filtering, and macro/pooled metrics.
- Existing HCP cache is required and fingerprinted. Copy the Original cache;
  silently regenerating it changes forward/inverse contents in this repository.
- Optional --run-dir provides config/source/data/cache hashes, model.pt,
  diffusion.pt, and atomic latest.pt stage checkpoints; --resume restores matched
  weights, Adam and RNG. --final-validation adds an explicit optional report.

These fixes can affect scores relative to a454564, especially deterministic
validation and Adam restoration. Compare QCDR against **the same corrected
code with --qcdr-mode original**; do not silently reuse an old score as its control.
No training or GPU verification was performed during branch preparation.

Unmodified scientific choices: original per-day 80/10/10 split after concatenating
source files; random historical snapshots and subject-filtered HCP; diffusion
objective and sampling schedule; original replay CE terms; shared scalar alpha;
static constraint and decoder. Original diffusion indexes relation conditions
through its entity table. That inherited semantic concern is documented rather
than silently changed as part of this method. Supported datasets must have enough
entity rows for all relation IDs, including inverses.

The original commit tracked datasets, model files, caches and logs. This branch
removes them from the tracked tree, not from the old working directory or Git
history. Supply datasets and the matching history_snap_v2 outside Git using
--data-root. Runtime artifacts are ignored.

## QCDR branch provenance

`codex/qcdr` starts at `f7bf2d8`, a direct child of `a454564` containing the
common runtime fixes above. No prior experimental method commit is an ancestor.
Three method-independent fixes previously mixed into `918573f` are retained
separately: keep diffusion CE targets one-dimensional for a singleton batch;
validate each existing HCP dictionary against its own training snapshot (accept
the supplied forward-only or forward+inverse convention without changing it);
pass only the train history into final evaluation (the appended validation
suffix was unused, but obscured the intended input). No gate, selector, KD,
feature statistics, or earlier experiment/diagnostic code is copied.
