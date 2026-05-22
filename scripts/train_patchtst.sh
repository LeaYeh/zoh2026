#!/usr/bin/env bash
set -e

EXP=${1:-"patchtst_v0"}
CONFIG="configs/exp/${EXP}.yaml"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "Training PatchTST: $EXP"
python -m src.forecasting.patchtst_finetune --config "$CONFIG" --run-name "$EXP"
