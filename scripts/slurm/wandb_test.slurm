#!/bin/bash
#SBATCH --job-name=zoh-wandb-test
#SBATCH --partition=boost_usr_prod
#SBATCH --account=PLACEHOLDER_ACCOUNT   # <-- fill in competition account code
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# WandB connectivity smoke test — 100 steps, tiny model, ~2 min on A100.
#
# Usage:
#   sbatch scripts/slurm/wandb_test.slurm
#
# After the job completes, run from the LOGIN NODE:
#   wandb sync $WORK/zoh2026/wandb/offline-run-*/
# Then check https://wandb.ai for the "leonardo_wandb_test" run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="${WORK:-$HOME}/zoh2026"
CONFIG="configs/exp/leonardo_wandb_test.yaml"

echo "==> Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "==> Config: $CONFIG"

cd "$REPO_ROOT"

# Load .env for WANDB_PROJECT / WANDB_ENTITY / WANDB_API_KEY
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# Compute nodes have no internet — write locally, sync from login node after
export WANDB_MODE=offline
export WANDB_DIR="$WORK_DIR/wandb"
mkdir -p "$WANDB_DIR"

export PATH="$HOME/.cargo/bin:$PATH"

echo "==> WANDB_MODE=$WANDB_MODE  WANDB_DIR=$WANDB_DIR"
echo "==> Starting training..."

uv run python src/train.py "$CONFIG"

echo ""
echo "==> Done. To push logs to WandB cloud, run from the LOGIN NODE:"
echo "    wandb sync $WORK_DIR/wandb/offline-run-*/"
