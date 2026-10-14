#!/usr/bin/env bash
# Foreground group runner; detach/sequentialize in the outer setsid shell.
set -euo pipefail
DCR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DCR_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
  DCR_DRY_RUN=true
  shift
fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_dcr_experiment.sh [--dry-run] original|mean|x0|guided-x0 [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
DCR_MODE=$1
DCR_GPU=${2:-0}
DCR_DATASET=${3:-ICEWS14s}
DCR_SEED=${4:-1}
DCR_PYTHON=${DGAR_PYTHON:-python}
DCR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$DCR_MODE" in
  original|mean|x0|guided-x0) ;;
  *) printf 'Unknown group: %s\n' "$DCR_MODE" >&2; exit 2 ;;
esac
if [[ ! $DCR_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2
  exit 2
fi
cd "$DCR_ROOT"
DCR_RUN_ID="$DCR_DATASET-dcr-$DCR_MODE-seed$DCR_SEED-$(date +%Y%m%d_%H%M%S)_$$"
DCR_RUN_DIR="$DCR_ROOT/experiment_runs/$DCR_RUN_ID"
DCR_LOG="$DCR_ROOT/logs/$DCR_RUN_ID.log"
DCR_COMMAND=("$DCR_PYTHON" -u "$DCR_ROOT/src/main.py" --gpu "$DCR_GPU" --dataset "$DCR_DATASET"
  --seed "$DCR_SEED" --data-root "$DCR_DATA_ROOT" --run-dir "$DCR_RUN_DIR"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation --dcr-mode "$DCR_MODE")
CUDA_VISIBLE_DEVICES='' "$DCR_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${DCR_COMMAND[@]:3}"
printf 'Log: %s\nRun: %s\nCommand:' "$DCR_LOG" "$DCR_RUN_DIR"
printf ' %q' "${DCR_COMMAND[@]}"
printf '\n'
if "$DCR_DRY_RUN"; then
  exit 0
fi
mkdir -p "$DCR_ROOT/logs" "$DCR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${DCR_COMMAND[@]}" 2>&1 | tee "$DCR_LOG"
