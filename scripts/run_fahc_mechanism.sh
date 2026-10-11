#!/usr/bin/env bash
# A bounded random-replay trajectory with temporary mechanism measurements.
set -euo pipefail
dry=()
if [[ ${1:-} == --dry-run ]]; then
    dry=(--dry-run)
    shift
fi
if (( $# > 4 )); then
    echo "Usage: bash scripts/run_fahc_mechanism.sh [--dry-run] GPU [DATASET] [SEED] [STAGES]" >&2
    exit 2
fi
gpu=${1:-0}
dataset=${2:-ICEWS14s}
seed=${3:-1}
stages=${4:-60,90,120}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$script_dir/run_fahc_experiment.sh" "${dry[@]}" \
    Mechanism "$gpu" "$dataset" "$seed" random 4096 0.1 0 \
    --fahc-eval-every 0 --fahc-mechanism-stages "$stages" \
    --fahc-mechanism-reference-steps 10
