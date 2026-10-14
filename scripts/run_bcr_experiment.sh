#!/usr/bin/env bash
# Foreground group runner; detach/sequentialize in the outer setsid shell.
set -euo pipefail
BCR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BCR_DRY_RUN=false
if [[ ${1:-} == --dry-run ]]; then
  BCR_DRY_RUN=true
  shift
fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_bcr_experiment.sh [--dry-run] original|outgoing|incident|random [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
BCR_MODE=$1
BCR_GPU=${2:-0}
BCR_DATASET=${3:-ICEWS14s}
BCR_SEED=${4:-1}
BCR_PYTHON=${DGAR_PYTHON:-python}
BCR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$BCR_MODE" in
  original|outgoing|incident|random) ;;
  *) printf 'Unknown group: %s\n' "$BCR_MODE" >&2; exit 2 ;;
esac
if [[ ! $BCR_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2
  exit 2
fi
cd "$BCR_ROOT"
BCR_RUN_ID="$BCR_DATASET-bcr-$BCR_MODE-seed$BCR_SEED-$(date +%Y%m%d_%H%M%S)_$$"
BCR_RUN_DIR="$BCR_ROOT/experiment_runs/$BCR_RUN_ID"
BCR_LOG="$BCR_ROOT/logs/$BCR_RUN_ID.log"
BCR_COMMAND=("$BCR_PYTHON" -u "$BCR_ROOT/src/main.py" --gpu "$BCR_GPU" --dataset "$BCR_DATASET"
  --seed "$BCR_SEED" --data-root "$BCR_DATA_ROOT" --run-dir "$BCR_RUN_DIR"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation --bcr-mode "$BCR_MODE")
CUDA_VISIBLE_DEVICES='' "$BCR_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${BCR_COMMAND[@]:3}"
printf 'Log: %s\nRun: %s\nCommand:' "$BCR_LOG" "$BCR_RUN_DIR"
printf ' %q' "${BCR_COMMAND[@]}"
printf '\n'
if "$BCR_DRY_RUN"; then
  exit 0
fi
mkdir -p "$BCR_ROOT/logs" "$BCR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${BCR_COMMAND[@]}" 2>&1 | tee "$BCR_LOG"
