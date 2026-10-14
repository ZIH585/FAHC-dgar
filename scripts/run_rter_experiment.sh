#!/usr/bin/env bash
# Optional end-to-end training. RTER itself only changes final entity scoring.
set -euo pipefail
RTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RTER_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then RTER_DRY_RUN=true; shift; fi
if [[ $# -lt 1 || $# -gt 5 ]]; then
  printf 'Usage: bash scripts/run_rter_experiment.sh [--dry-run] original|transition|repeat|popularity|shuffled [GPU=0] [DATASET=ICEWS14s] [SEED=1] [BETA=0.1]\n' >&2
  exit 2
fi
RTER_MODE=$1
RTER_GPU=${2:-0}
RTER_DATASET=${3:-ICEWS14s}
RTER_SEED=${4:-1}
RTER_BETA=${5:-0.1}
RTER_PYTHON=${DGAR_PYTHON:-python}
RTER_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$RTER_MODE" in original|transition|repeat|popularity|shuffled) ;; *) exit 2 ;; esac
if [[ ! $RTER_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then printf 'Invalid dataset\n' >&2; exit 2; fi
cd "$RTER_ROOT"
RTER_ID="$RTER_DATASET-rter-$RTER_MODE-seed$RTER_SEED-$(date +%Y%m%d_%H%M%S)_$$"
RTER_RUN="$RTER_ROOT/experiment_runs/$RTER_ID"
RTER_LOG="$RTER_ROOT/logs/$RTER_ID.log"
RTER_COMMAND=("$RTER_PYTHON" -u "$RTER_ROOT/src/main.py" --gpu "$RTER_GPU" --dataset "$RTER_DATASET"
  --seed "$RTER_SEED" --data-root "$RTER_DATA_ROOT" --run-dir "$RTER_RUN" --final-validation
  --reply_batch 35 --mu 1 --mu_r 1 --rter-mode "$RTER_MODE" --rter-beta "$RTER_BETA")
CUDA_VISIBLE_DEVICES='' "$RTER_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser,validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${RTER_COMMAND[@]:3}"
printf 'Log: %s\nRun: %s\nCommand:' "$RTER_LOG" "$RTER_RUN"
printf ' %q' "${RTER_COMMAND[@]}"
printf '\n'
if "$RTER_DRY_RUN"; then exit 0; fi
mkdir -p "$RTER_ROOT/logs" "$RTER_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${RTER_COMMAND[@]}" 2>&1 | tee "$RTER_LOG"
