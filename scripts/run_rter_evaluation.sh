#!/usr/bin/env bash
# Recommended paired screen: reuse completed Original, never optimize weights.
set -euo pipefail
RTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RTER_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then RTER_DRY_RUN=true; shift; fi
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_rter_evaluation.sh [--dry-run] validation|test GPU ORIGINAL_RUN [BETAS_or_VALIDATION_SUMMARY]\n' >&2
  exit 2
fi
RTER_SPLIT=$1
RTER_GPU=$2
RTER_INPUT=$3
case "$RTER_SPLIT" in
 validation) RTER_BETAS=${4:-0.1,0.3,1}; RTER_MODES=${RTER_MODES:-transition,repeat,popularity,shuffled} ;;
 test) RTER_BETAS=${4:?TEST requires the beta selected on VALIDATION}; RTER_MODES=${RTER_MODES:-transition} ;;
 *) exit 2 ;;
esac
RTER_PYTHON=${DGAR_PYTHON:-python}
RTER_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
cd "$RTER_ROOT"
RTER_ID="rter-$RTER_SPLIT-$(date +%Y%m%d_%H%M%S)_$$"
RTER_OUTPUT="$RTER_ROOT/experiment_runs/$RTER_ID"
RTER_LOG="$RTER_ROOT/logs/$RTER_ID.log"
RTER_EXTRA=()
if [[ $RTER_SPLIT == test && $RTER_BETAS == *.json ]]; then
  RTER_EXTRA=(--selection-file "$RTER_BETAS")
  RTER_BETAS=0.1 # Replaced by the locked value from the summary before scoring.
fi
RTER_COMMAND=("$RTER_PYTHON" -u "$RTER_ROOT/scripts/evaluate_rter.py" --run-dir "$RTER_INPUT"
  --output "$RTER_OUTPUT" --gpu "$RTER_GPU" --data-root "$RTER_DATA_ROOT" --split "$RTER_SPLIT"
  --stages all --betas "$RTER_BETAS" --modes "$RTER_MODES" "${RTER_EXTRA[@]}")
printf 'Log: %s\nOutput: %s\nCommand:' "$RTER_LOG" "$RTER_OUTPUT"
printf ' %q' "${RTER_COMMAND[@]}"
printf '\n'
if "$RTER_DRY_RUN"; then exit 0; fi
mkdir -p "$RTER_ROOT/logs" "$RTER_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${RTER_COMMAND[@]}" 2>&1 | tee "$RTER_LOG"
