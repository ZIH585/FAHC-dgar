#!/usr/bin/env bash
# Foreground runner: use setsid outside this script to detach a single/paired run.
set -euo pipefail
RAHA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAHA_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
  RAHA_DRY_RUN=true
  shift
fi
if [[ $# -lt 1 || $# -gt 5 ]]; then
  printf 'Usage: bash scripts/run_raha_experiment.sh [--dry-run] original|shared|role [GPU=0] [DATASET=ICEWS14s] [SEED=1] [RANK=shared:16/role:8]\n' >&2
  exit 2
fi
RAHA_MODE=$1
RAHA_GPU=${2:-0}
RAHA_DATASET=${3:-ICEWS14s}
RAHA_SEED=${4:-1}
RAHA_PYTHON=${DGAR_PYTHON:-python}
RAHA_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$RAHA_MODE" in
  shared) RAHA_RANK=${5:-16} ;;
  original|role) RAHA_RANK=${5:-8} ;;
  *) printf 'Unknown group: %s\n' "$RAHA_MODE" >&2; exit 2 ;;
esac
if [[ ! $RAHA_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2
  exit 2
fi
cd "$RAHA_ROOT"
RAHA_RUN_ID="$RAHA_DATASET-raha-$RAHA_MODE-r$RAHA_RANK-seed$RAHA_SEED-$(date +%Y%m%d_%H%M%S)_$$"
RAHA_RUN_DIR="$RAHA_ROOT/experiment_runs/$RAHA_RUN_ID"
RAHA_LOG="$RAHA_ROOT/logs/$RAHA_RUN_ID.log"
RAHA_COMMAND=("$RAHA_PYTHON" -u "$RAHA_ROOT/src/main.py"
  --gpu "$RAHA_GPU" --dataset "$RAHA_DATASET" --seed "$RAHA_SEED"
  --data-root "$RAHA_DATA_ROOT" --run-dir "$RAHA_RUN_DIR"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation
  --raha-mode "$RAHA_MODE" --raha-rank "$RAHA_RANK" --raha-diagnostics)
CUDA_VISIBLE_DEVICES='' "$RAHA_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${RAHA_COMMAND[@]:3}"
printf 'Log: %s\nRun: %s\nCommand:' "$RAHA_LOG" "$RAHA_RUN_DIR"
printf ' %q' "${RAHA_COMMAND[@]}"
printf '\n'
if "$RAHA_DRY_RUN"; then
  exit 0
fi
mkdir -p "$RAHA_ROOT/logs" "$RAHA_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${RAHA_COMMAND[@]}" 2>&1 | tee "$RAHA_LOG"
