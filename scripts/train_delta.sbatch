#!/usr/bin/env bash
# Submit from the repository root.
#SBATCH --job-name=k1-goalkeeper
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --account=bign-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f run.py || ! -f vendor/booster_mjlab/pyproject.toml ]]; then
  echo "ERROR: submit from the goalkeeping repository root" >&2
  exit 1
fi

mkdir -p logs/rsl_rl
export WANDB_DIR="${SLURM_SUBMIT_DIR}/logs"

uv run --project vendor/booster_mjlab --locked --no-dev \
  python run.py train Goalkeeper-K1-Sim \
  --log-root "${SLURM_SUBMIT_DIR}/logs/rsl_rl" "$@"
