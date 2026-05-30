#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=0:30:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --account=EUHPC_D30_031

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
pixi run python src/train.py "$CONFIG"

echo "==> Training done. Sync WandB from login node:"
echo "    wandb sync $WORK_DIR/wandb/offline-run-*/"
