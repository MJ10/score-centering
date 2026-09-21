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

# cluv points the job's stdout at <results_path>/<cluster>_<jobid>/slurm-<jobid>.out
# (via --output in cluv >= 0.1, via $SBATCH_OUTPUT in older versions). The run
# directory is that file's parent.
output_path="$(scontrol show job "${SLURM_JOB_ID:?not running under Slurm}" -o 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -n 1 || true)"
output_path="${output_path:-${SBATCH_OUTPUT:-}}"
if [[ -z "$output_path" ]]; then
    echo "Could not determine the job's output path; submit this script through cluv." >&2
    exit 2
fi
run_dir="${output_path%/*}"
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

# A node can come up with fewer visible GPUs than Slurm allocated (fir
# fc10501, 2026-09-21: 3 of 4). Fail fast instead of training on a lopsided
# mesh; the job will be requeued elsewhere by a resubmit.
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]] && command -v nvidia-smi >/dev/null; then
    visible=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " ")
    if (( visible < SLURM_GPUS_ON_NODE )); then
        echo "Only $visible of $SLURM_GPUS_ON_NODE allocated GPUs are visible on $(hostname); aborting." >&2
        exit 3
    fi
fi

echo "GIT_COMMIT=${GIT_COMMIT:-unknown}"
echo "Run directory: $run_dir"
printf 'Running command:'
printf ' %q' "${command[@]}"
printf '\n'

srun uv run --no-sync "${command[@]}"
