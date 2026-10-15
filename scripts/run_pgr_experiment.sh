#!/usr/bin/env bash
# One foreground experiment; detach/serialize externally with setsid.
set -euo pipefail
PGR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PGR_DRY=false
if [[ ${1:-} == --dry-run ]]; then PGR_DRY=true; shift; fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_pgr_experiment.sh [--dry-run] original|mean|shared|mean-shared|geometry|geometry-random|geometry-isotropic [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
PGR_MODE=$1
PGR_GPU=${2:-0}
PGR_DATASET=${3:-ICEWS14s}
PGR_SEED=${4:-1}
PGR_PYTHON=${DGAR_PYTHON:-python}
PGR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$PGR_MODE" in
  original|mean|shared|mean-shared|geometry|geometry-random|geometry-isotropic) ;;
  *) printf 'Unknown group: %s\n' "$PGR_MODE" >&2; exit 2 ;;
esac
if [[ ! $PGR_DATASET =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'Invalid dataset name\n' >&2; exit 2
fi
cd "$PGR_ROOT"
PGR_ID="$PGR_DATASET-pgr-$PGR_MODE-seed$PGR_SEED-$(date +%Y%m%d_%H%M%S)_$$"
PGR_COMMAND=("$PGR_PYTHON" -u "$PGR_ROOT/src/main.py"
  --gpu "$PGR_GPU" --dataset "$PGR_DATASET" --seed "$PGR_SEED"
  --data-root "$PGR_DATA_ROOT" --run-dir "$PGR_ROOT/experiment_runs/$PGR_ID"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation
  --pgr-mode "$PGR_MODE" --pgr-adapter-rank 16 --pgr-risk-rank 4
  --pgr-geometry-facts 64 --pgr-risk-strength 1 --pgr-diagnostics)
# Validation only: never load datasets, initialize models, or touch CUDA here.
CUDA_VISIBLE_DEVICES='' "$PGR_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' \
  "${PGR_COMMAND[@]:3}"
printf 'Run: %s\nLog: %s\nCommand:' "$PGR_ROOT/experiment_runs/$PGR_ID" "$PGR_ROOT/logs/$PGR_ID.log"
printf ' %q' "${PGR_COMMAND[@]}"
printf '\n'
if "$PGR_DRY"; then exit 0; fi
mkdir -p "$PGR_ROOT/logs" "$PGR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PGR_COMMAND[@]}" 2>&1 | tee "$PGR_ROOT/logs/$PGR_ID.log"
