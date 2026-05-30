#!/usr/bin/env bash
# One-time setup on Leonardo login node.
# Run once after cloning the repo:
#   bash scripts/setup_leonardo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Repo root: $REPO_ROOT"

# ── 1. Install uv (user-level, no root needed) ───────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "==> uv already installed: $(uv --version)"
fi

# ── 2. Create virtualenv and install deps ────────────────────────────────────
cd "$REPO_ROOT"
echo "==> Creating .venv and installing dependencies..."
uv sync

# ── 3. Create .env from example if not present ───────────────────────────────
if [[ ! -f "$REPO_ROOT/.env" ]]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "==> Created .env — fill in your keys:"
    echo "      WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY"
else
    echo "==> .env already exists, skipping"
fi

# ── 4. Create persistent directories on \$WORK ────────────────────────────────
WORK_DIR="${WORK:-$HOME}/zoh2026"
mkdir -p "$WORK_DIR/wandb"
mkdir -p "$WORK_DIR/checkpoints"
mkdir -p "$WORK_DIR/logs"
echo "==> Work directories created under $WORK_DIR"

# ── 5. Smoke test (CPU only, no WandB) ───────────────────────────────────────
echo "==> Running smoke test (200 steps, no GPU, no WandB)..."
WANDB_MODE=disabled uv run python src/train.py configs/exp/gpt2_dummy.yaml
echo "==> Smoke test passed"

echo ""
echo "====================================================="
echo " Setup complete. Next steps:"
echo "  1. Fill in .env (WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY)"
echo "  2. Set SLURM_ACCOUNT in scripts/slurm/train.slurm"
echo "  3. Submit: sbatch scripts/slurm/train.slurm"
echo "====================================================="
