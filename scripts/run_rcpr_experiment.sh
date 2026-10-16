#!/usr/bin/env bash
# One foreground experiment. Use setsid externally and choose GPU explicitly.
set -euo pipefail
RCPR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RCPR_DRY=false
if [[ ${1:-} == --dry-run ]]; then RCPR_DRY=true; shift; fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_rcpr_experiment.sh [--dry-run] original|mean-shared|mean-head|prototype|shuffled [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
RCPR_MODE=$1
RCPR_GPU=${2:-0}
RCPR_DATASET=${3:-ICEWS14s}
RCPR_SEED=${4:-1}
RCPR_PYTHON=${DGAR_PYTHON:-python}
RCPR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$RCPR_MODE" in
  original|mean-shared|mean-head|prototype|shuffled) ;;
  *) printf 'Unknown mode: %s\n' "$RCPR_MODE" >&2; exit 2 ;;
esac
if [[ ! $RCPR_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2; exit 2
fi
cd "$RCPR_ROOT"
RCPR_ID="$RCPR_DATASET-rcpr-$RCPR_MODE-seed$RCPR_SEED-$(date +%Y%m%d_%H%M%S)_$$"
RCPR_COMMAND=("$RCPR_PYTHON" -u "$RCPR_ROOT/src/main.py"
  --gpu "$RCPR_GPU" --dataset "$RCPR_DATASET" --seed "$RCPR_SEED"
  --data-root "$RCPR_DATA_ROOT" --run-dir "$RCPR_ROOT/experiment_runs/$RCPR_ID"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation
  --rcpr-mode "$RCPR_MODE" --rcpr-adapter-rank 16 --rcpr-rank 16
  --rcpr-score-chunk 256 --rcpr-shuffle-seed 0 --rcpr-diagnostics)
CUDA_VISIBLE_DEVICES='' "$RCPR_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser,validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${RCPR_COMMAND[@]:3}"
printf 'Run: %s\nLog: %s\nCommand:' "$RCPR_ROOT/experiment_runs/$RCPR_ID" "$RCPR_ROOT/logs/$RCPR_ID.log"
printf ' %q' "${RCPR_COMMAND[@]}"
printf '\n'
if "$RCPR_DRY"; then exit 0; fi
mkdir -p "$RCPR_ROOT/logs" "$RCPR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${RCPR_COMMAND[@]}" 2>&1 | tee "$RCPR_ROOT/logs/$RCPR_ID.log"
