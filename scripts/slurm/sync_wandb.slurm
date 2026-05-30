#!/bin/bash
#SBATCH --job-name=wandb-sync
#SBATCH --partition=lrd_all_serial
#SBATCH --account=PLACEHOLDER_ACCOUNT   # <-- same as train.slurm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=wandb_sync_%j.out

# Syncs all pending offline WandB runs to the dashboard.
# Automatically chained after training:
#   TRAIN_JOB=$(sbatch scripts/slurm/train.slurm | awk '{print $4}')
#   sbatch --dependency=afterok:$TRAIN_JOB scripts/slurm/sync_wandb.slurm

set -euo pipefail

WORK_DIR="${WORK:-$HOME}/zoh2026"

if [[ -f "$HOME/project/git_dev/side_project/ai-competition/.env" ]]; then
    set -a; source "$HOME/project/git_dev/side_project/ai-competition/.env"; set +a
fi

export PATH="$HOME/.cargo/bin:$PATH"

echo "==> Syncing offline WandB runs from $WORK_DIR/wandb/"
wandb sync --no-include-synced "$WORK_DIR/wandb/offline-run-"*/
echo "==> Sync complete"
