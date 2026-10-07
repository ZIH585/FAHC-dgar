#!/usr/bin/env bash
# One experiment in an isolated checkout-free output directory.
set -euo pipefail

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
if (( $# < 8 )); then
    echo "Usage: bash scripts/run_fahc_experiment.sh [--dry-run] LABEL GPU DATASET SEED MODE BUDGET ETA KD [extra main.py flags]" >&2
    exit 2
fi
label=$1
gpu=$2
dataset=$3
seed=$4
mode=$5
budget=$6
eta=$7
kd=$8
shift 8
for extra in "$@"; do
    case "$extra" in
        --gpu|--gpu=*|-d|--dataset|--dataset=*|--seed|--seed=*|--hcp-selection|--hcp-selection=*|--fahc-fact-budget|--fahc-fact-budget=*|--fahc-eta|--fahc-eta=*|--fahc-kd-weight|--fahc-kd-weight=*|--fahc-log|--fahc-log=*)
            echo "Set $extra through the positional arguments; the log path is managed by the launcher" >&2
            exit 2
            ;;
        --test|--grid-search)
            echo "The experiment launcher runs one chronological training configuration" >&2
            exit 2
            ;;
    esac
done
[[ $label =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Use letters, digits, _ or - in LABEL" >&2; exit 2; }
[[ $dataset =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid dataset name" >&2; exit 2; }
[[ $gpu =~ ^[0-9]+$ ]] || { echo "GPU must be a nonnegative integer" >&2; exit 2; }
[[ $seed =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${DGAR_PYTHON:-/data1/zihao/miniconda3/envs/dgar_env/bin/python}
run_root=${DGAR_RUN_ROOT:-$repo_root/experiment_runs}
[[ $run_root == /* ]] || { echo "DGAR_RUN_ROOT must be an absolute path" >&2; exit 2; }
[[ -x $python_bin ]] || { echo "Python not found: $python_bin" >&2; exit 2; }

run_dir=$run_root/${dataset}_${label}_s${seed}_$(date +%Y%m%d_%H%M%S_%N)_$$
cli=(
    --gpu "$gpu" --dataset "$dataset" --seed "$seed"
    --hcp-selection "$mode" --reply_batch 35 --fahc-fact-budget "$budget"
    --fahc-eta "$eta" --fahc-kd-weight "$kd"
    --n-epochs 1 --regcn_epochs 10 --diffu_epochs 10
    --lr 0.001 --task-weight 0.7 --mu 1 --mu_r 1
    --n-hidden 200 --hidden_size 200
    --train-history-len 3 --test-history-len 3
    --fahc-candidate-multiplier 2 --fahc-entity-weight 1
    --fahc-probe-steps 1 --fahc-probe-facts 128
    --fahc-score-batch-size 256 --fahc-kd-temperature 2 --fahc-kd-confidence 0.5
    --fahc-eval-every 1000000000 --fahc-past-eval-samples 0
    --fahc-diagnostics --fahc-final-validation --fahc-log "$run_dir/logs/stages.jsonl"
    "$@"
)
# This parses only the CLI AST and reads package metadata; no model/CUDA import.
config=$(PYTHONDONTWRITEBYTECODE=1 "$python_bin" "$script_dir/fahc_run_config.py" "${cli[@]}")
if (( dry_run )); then
    printf 'Validated. Output directory: %s\n' "$run_dir"
    printf 'Command: '
    printf '%q ' "$python_bin" -u "$repo_root/src/main.py" "${cli[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$run_root/.locks"
exec 9> "$run_root/.locks/gpu${gpu}.lock"
flock -n 9 || { echo "Another experiment launcher is using GPU $gpu" >&2; exit 1; }
# Prevent main.py's CPU fallback from silently consuming a long experiment.
"$python_bin" -c 'import sys, torch; index = int(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU training"; assert index < torch.cuda.device_count(), "GPU index outside visible devices"' "$gpu"
mkdir "$run_dir"
mkdir -p "$run_dir/src/logs" "$run_dir/models" "$run_dir/diffu_models" "$run_dir/logs"
ln -s "$repo_root/data" "$run_dir/data"
printf '%s\n' "$config" > "$run_dir/config.json"
printf '%q ' "$python_bin" -u "$repo_root/src/main.py" "${cli[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf 'RUN_DIR=%s\n' "$run_dir"
cd -- "$run_dir/src"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONDONTWRITEBYTECODE=1
started=$(date +%s)
set +e
"$python_bin" -u "$repo_root/src/main.py" "${cli[@]}" 2>&1 | tee "$run_dir/logs/train.log"
codes=("${PIPESTATUS[@]}")
set -e
finished=$(date +%s)
printf '{"python_exit_code":%d,"tee_exit_code":%d,"wall_seconds":%d}\n' \
    "${codes[0]}" "${codes[1]}" "$((finished-started))" > "$run_dir/status.json"
if (( codes[0] != 0 || codes[1] != 0 )); then
    printf 'Experiment failed. Check %s\n' "$run_dir/logs/train.log" >&2
    exit 1
fi
printf 'Completed: %s\n' "$run_dir"
