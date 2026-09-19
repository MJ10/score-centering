#!/bin/bash
# Slurm job script for `cluv submit`. Resources come from [tool.cluv.sbatch_args]
# in pyproject.toml and from flags passed before `--`:
#
#   cluv submit mila -- python train_rl.py model.source=Qwen/Qwen3-0.6B rl.sc=true
#   cluv submit fir --time=12:00:00 --gpus-per-node=h100:4 -- python train_rl.py ...
#   cluv submit mila -- wandb agent <entity>/score-centering/<sweep-id>
#
# For train_*.py commands the script routes outputs to this job's results
# directory and the weight cache to $POSTAX_WEIGHTS_DIR unless the command
# already sets log.dir / model.weights_dir / log.wandb_mode.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: cluv submit <cluster> -- python train_rl.py [key=value ...]" >&2
    exit 2
fi

# cluv sets SBATCH_OUTPUT to <results_path>/<cluster>_%j/slurm-%j.out.
output_pattern="${SBATCH_OUTPUT:?This job script must be submitted through cluv}"
run_dir="${output_pattern%/*}"
run_dir="${run_dir//%j/${SLURM_JOB_ID}}"
run_dir="${run_dir//%A/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
run_dir="${run_dir//%a/${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$run_dir"

# Used by the W&B sweep configs in sweeps/ (`cd $POSTAX_DIR && python train_rl.py ...`).
export POSTAX_DIR="$PWD"
export POSTAX_WEIGHTS_DIR="${POSTAX_WEIGHTS_DIR:-${SCRATCH:+$SCRATCH/postax/weights}}"
export POSTAX_WEIGHTS_DIR="${POSTAX_WEIGHTS_DIR:-$HOME/.cache/postax/weights}"
mkdir -p "$POSTAX_WEIGHTS_DIR"

has_override() {
    local key="$1"; shift
    local arg
    for arg in "$@"; do
        [[ "$arg" == "$key="* ]] && return 0
    done
    return 1
}

command=("$@")
if [[ "${command[0]}" == python* && "${command[1]:-}" == train_*.py ]]; then
    has_override log.dir "${command[@]}" || command+=("log.dir=$run_dir")
    has_override model.weights_dir "${command[@]}" || command+=("model.weights_dir=$POSTAX_WEIGHTS_DIR")
    # utils.parse_cfg passes mode= explicitly to wandb.init, which ignores $WANDB_MODE.
    if [[ -n "${WANDB_MODE:-}" ]]; then
        has_override log.wandb_mode "${command[@]}" || command+=("log.wandb_mode=$WANDB_MODE")
    fi
fi

echo "GIT_COMMIT=${GIT_COMMIT:-unknown}"
echo "Run directory: $run_dir"
printf 'Running command:'
printf ' %q' "${command[@]}"
printf '\n'

srun uv run --no-sync "${command[@]}"
