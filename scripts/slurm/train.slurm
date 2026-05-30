#!/bin/bash
#SBATCH --job-name=zoh-gpt2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=PLACEHOLDER_ACCOUNT   # <-- fill in competition account code
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch scripts/slurm/train.slurm
#   sbatch scripts/slurm/train.slurm configs/exp/gpt2_finetune_v1.yaml

set -euo pipefail

CONFIG="${1:-configs/exp/gpt2_finetune_v1.yaml}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK_DIR="${WORK:-$HOME}/zoh2026"

echo "==> Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "==> Config: $CONFIG"
echo "==> Repo:   $REPO_ROOT"

cd "$REPO_ROOT"

# ── env ──────────────────────────────────────────────────────────────────────
# Load .env for WANDB_PROJECT / WANDB_ENTITY / WANDB_API_KEY
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# Compute nodes have no internet — use offline mode, sync later from login node
export WANDB_MODE=offline
export WANDB_DIR="$WORK_DIR/wandb"

export PATH="$HOME/.cargo/bin:$PATH"   # uv lives here

echo "==> WANDB_MODE=$WANDB_MODE  WANDB_DIR=$WANDB_DIR"

# ── training ─────────────────────────────────────────────────────────────────
uv run python src/train.py "$CONFIG"

echo "==> Training done. Sync WandB from login node:"
echo "    wandb sync $WORK_DIR/wandb/offline-run-*/"
