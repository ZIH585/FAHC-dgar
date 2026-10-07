#!/usr/bin/env bash
# These stages are separate deliberately: inspect validation before continuing.
set -euo pipefail
stage=${1:-}
gpu=${2:-0}
dataset=${3:-ICEWS14s}
seed=${4:-1}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run() {
    bash "$script_dir/run_fahc_experiment.sh" "$1" "$gpu" "$dataset" "$seed" "$2" "$3" "$4" "$5"
}
case "$stage" in
    core)
        run DGAR original 4096 0.1 0
        run R4096 random 4096 0.1 0
        run A4096_E01 risk+coverage 4096 0.1 0
        ;;
    budget)
        run R128 random 128 0.1 0
        run A128_E01 risk+coverage 128 0.1 0
        run R1024 random 1024 0.1 0
        run A1024_E01 risk+coverage 1024 0.1 0
        ;;
    selection)
        run A4096_E1 risk+coverage 4096 1 0
        run Risk4096 risk 4096 0.1 0
        run Coverage4096 coverage 4096 0.1 0
        run Recent4096 recent 4096 0.1 0
        ;;
    kd)
        run R4096_KD003 random 4096 0.1 0.03
        run A4096_KD003 risk+coverage 4096 0.1 0.03
        ;;
    *)
        echo "Usage: bash scripts/run_fahc_stage.sh core|budget|selection|kd GPU [DATASET] [SEED]" >&2
        exit 2
        ;;
esac
