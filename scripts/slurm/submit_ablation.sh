#!/usr/bin/env bash
# Submit all 3 Leonardo ablation jobs and chain a WandB sync after each.
#
# Usage:
#   bash scripts/slurm/submit_ablation.sh
#
# After all jobs finish, sync_wandb.slurm pushes each run's offline logs to WandB.
# Check progress at https://wandb.ai under group "leonardo-ablation".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_wandb.slurm"
TRAIN_SCRIPT="$SCRIPT_DIR/train.slurm"

CONFIGS=(
    "configs/exp/leonardo_A_family_token.yaml"
    "configs/exp/leonardo_B_desc_embed.yaml"
    "configs/exp/leonardo_C_desc_categ.yaml"
)

echo "==> Submitting ${#CONFIGS[@]} ablation jobs"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    RUN_NAME=$(basename "$CONFIG" .yaml)

    TRAIN_JID=$(sbatch --parsable "$TRAIN_SCRIPT" "$CONFIG")
    SYNC_JID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JID" "$SYNC_SCRIPT")

    echo "  [$RUN_NAME]"
    echo "    train job : $TRAIN_JID"
    echo "    sync  job : $SYNC_JID  (runs after train completes)"
done

echo ""
echo "==> All jobs queued. Monitor with:"
echo "    squeue -u \$USER"
echo "    # After sync jobs finish, check WandB group 'leonardo-ablation'"
