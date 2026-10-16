#!/usr/bin/env bash
set -euo pipefail
RCPR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RCPR_DRY=false
if [[ ${1:-} == --dry-run ]]; then RCPR_DRY=true; shift; fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf 'Usage: bash scripts/run_rcpr_diagnostic.sh [--dry-run] SOURCE_MEAN_SHARED_RUN [GPU=0] [STAGES=60,120,180,240,300]\n' >&2
  exit 2
fi
RCPR_SOURCE=$1
RCPR_GPU=${2:-0}
RCPR_STAGES=${3:-60,120,180,240,300}
RCPR_PYTHON=${DGAR_PYTHON:-python}
RCPR_DATA_ROOT=${DGAR_DATA_ROOT:-/data1/zihao/DGAR-code/data}
cd "$RCPR_ROOT"
RCPR_ID="rcpr-frozen-$(date +%Y%m%d_%H%M%S)_$$"
RCPR_COMMAND=("$RCPR_PYTHON" -u "$RCPR_ROOT/scripts/diagnose_rcpr.py"
  --run-dir "$RCPR_SOURCE" --output "$RCPR_ROOT/experiment_runs/$RCPR_ID"
  --data-root "$RCPR_DATA_ROOT" --gpu "$RCPR_GPU" --stages "$RCPR_STAGES"
  --facts 128 --steps 100 --lr .001 --rank 16 --score-chunk 256)
printf 'Run: %s\nLog: %s\nCommand:' "$RCPR_ROOT/experiment_runs/$RCPR_ID" "$RCPR_ROOT/logs/$RCPR_ID.log"
printf ' %q' "${RCPR_COMMAND[@]}"
printf '\n'
if "$RCPR_DRY"; then
  CUDA_VISIBLE_DEVICES='' "$RCPR_PYTHON" -B "$RCPR_ROOT/scripts/diagnose_rcpr.py" --help > /dev/null
  [[ -r "$RCPR_SOURCE/config.json" && -r "$RCPR_SOURCE/latest.pt" ]] || { printf 'Source run missing\n' >&2; exit 2; }
  exit 0
fi
mkdir -p "$RCPR_ROOT/logs" "$RCPR_ROOT/experiment_runs"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${RCPR_COMMAND[@]}" 2>&1 | tee "$RCPR_ROOT/logs/$RCPR_ID.log"
