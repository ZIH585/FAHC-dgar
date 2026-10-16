#!/usr/bin/env bash
# One experiment; detach/serialize externally. --dry-run never initializes CUDA.
set -euo pipefail
CUR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUR_DRY=false
if [[ ${1:-} == --dry-run ]]; then CUR_DRY=true; shift; fi
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf 'Usage: bash scripts/run_cur_experiment.sh [--dry-run] original|mean-shared|dual-replay|dual-direction|utility [GPU=0] [DATASET=ICEWS14s] [SEED=1]\n' >&2
  exit 2
fi
CUR_MODE=$1
CUR_GPU=${2:-0}
CUR_DATASET=${3:-ICEWS14s}
CUR_SEED=${4:-1}
CUR_PYTHON=${DGAR_PYTHON:-python}
CUR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
case "$CUR_MODE" in original|mean-shared|dual-replay|dual-direction|utility) ;; *) exit 2 ;; esac
[[ $CUR_DATASET =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
case ${DGAR_CPU_OFFLOAD:-0} in 0|1) ;; *) printf 'DGAR_CPU_OFFLOAD must be 0 or 1\n' >&2; exit 2 ;; esac
cd "$CUR_ROOT"
CUR_ID="$CUR_DATASET-cur-$CUR_MODE-seed$CUR_SEED-$(date +%Y%m%d_%H%M%S)_$$"
CUR_ARGS=(--gpu "$CUR_GPU" --dataset "$CUR_DATASET" --seed "$CUR_SEED"
  --data-root "$CUR_DATA_ROOT" --run-dir "$CUR_ROOT/experiment_runs/$CUR_ID"
  --mu 1 --mu_r 1 --reply_batch 35 --final-validation
  --cur-mode "$CUR_MODE" --cur-adapter-rank 16 --cur-router-hidden 32
  --cur-probe-facts 128 --cur-score-batch-size 256 --cur-router-steps 10
  --cur-router-lr 0.01 --cur-threshold 0.5 --cur-diagnostics)
CUDA_VISIBLE_DEVICES='' "$CUR_PYTHON" -B -c \
  'import sys; from src.cli import build_arg_parser, validate_args; validate_args(build_arg_parser().parse_args(sys.argv[1:]))' "${CUR_ARGS[@]}"
CUR_COMMAND=("$CUR_PYTHON" -u "$CUR_ROOT/scripts/run_cur_entry.py")
if [[ ${DGAR_CPU_OFFLOAD:-0} == 1 ]]; then CUR_COMMAND+=(--cpu-offload); fi
CUR_COMMAND+=("${CUR_ARGS[@]}")
printf 'Run: %s\nLog: %s\nCommand:' "$CUR_ROOT/experiment_runs/$CUR_ID" "$CUR_ROOT/logs/$CUR_ID.log"
printf ' %q' "${CUR_COMMAND[@]}"
printf '\n'
if "$CUR_DRY"; then exit 0; fi
mkdir -p "$CUR_ROOT/logs" "$CUR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CUR_COMMAND[@]}" 2>&1 | tee "$CUR_ROOT/logs/$CUR_ID.log"
