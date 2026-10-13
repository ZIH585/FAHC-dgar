#!/usr/bin/env bash
# Run one group in the foreground; the caller can detach it with setsid.
set -euo pipefail
GRB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRB_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
  GRB_DRY_RUN=true
  shift
fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_grb_experiment.sh [--dry-run] original|half|adaptive [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
GRB_GROUP=$1
GRB_GPU=${2:-0}
GRB_DATASET=${3:-ICEWS14s}
GRB_SEED=${4:-1}
GRB_PYTHON=${DGAR_PYTHON:-python}
GRB_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$GRB_GROUP" in
  original) GRB_OPTIONS=(--grb-mode original) ;;
  half) GRB_OPTIONS=(--grb-mode fixed --grb-fixed-weight 0.5) ;;
  adaptive) GRB_OPTIONS=(--grb-mode adaptive --grb-min-weight 0.5 --grb-target-ratio 1.0 --grb-conflict-scale 1.0 --grb-ema 0.8) ;;
  *) printf 'Unknown group: %s\n' "$GRB_GROUP" >&2; exit 2 ;;
esac
# Restrict dataset text before using it as part of an output filename.
if [[ ! $GRB_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2
  exit 2
fi
cd "$GRB_ROOT"
GRB_RUN_ID="$GRB_DATASET-$GRB_GROUP-seed$GRB_SEED-$(date +%Y%m%d_%H%M%S)_$$"
GRB_RUN_DIR="$GRB_ROOT/experiment_runs/$GRB_RUN_ID"
GRB_LOG="$GRB_ROOT/logs/$GRB_RUN_ID.log"
GRB_COMMAND=("$GRB_PYTHON" -u "$GRB_ROOT/src/main.py" --gpu "$GRB_GPU" --dataset "$GRB_DATASET"
  --seed "$GRB_SEED" --data-root "$GRB_DATA_ROOT" --run-dir "$GRB_RUN_DIR"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation "${GRB_OPTIONS[@]}")
# CPU argument validation only; neither data loading nor model construction.
CUDA_VISIBLE_DEVICES='' "$GRB_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${GRB_COMMAND[@]:3}"
printf 'Log: %s\nRun: %s\nCommand:' "$GRB_LOG" "$GRB_RUN_DIR"
printf ' %q' "${GRB_COMMAND[@]}"
printf '\n'
if "$GRB_DRY_RUN"; then
  exit 0
fi
mkdir -p "$GRB_ROOT/logs" "$GRB_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${GRB_COMMAND[@]}" 2>&1 | tee "$GRB_LOG"
