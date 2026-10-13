# DGAR baseline contract

This branch is the reference for future DGAR methods. Read docs/BASELINE.md
before changing the training or evaluation pipeline.

- New methods branch from codex/original-baseline and default to OFF. Verify
  the OFF path against this baseline before reporting method improvements.
- Preserve the pinned author's feature-generator TRAIN modes, weights-only
  rollback with last-update Adam state, shared training/validation RNG stream,
  HCP set iteration order, and retrospective final evaluation history flow.
  A change to any of these is an experimental protocol change; name/version it
  explicitly and do not present its OFF score as the unchanged Original.
- Keep necessary CPU/device/shape fixes narrow. Compare with the pinned source
  fixtures in tests/reference when a fix touches scientific behavior.
- Do not use paper numbers or older branch logs as the sole control for a new
  method. Use the same source, explicit seed policy, GPU/environment, data and
  HCP cache for Original and the method. Keep run directories separate.
- Preserve existing experiments. Do not overwrite datasets, caches, logs or
  checkpoints and do not add them to Git.
- Source changes require the CPU checks in scripts/check_baseline.sh. Do not
  start real training during code changes unless the user authorizes it.
