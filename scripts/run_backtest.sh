#!/usr/bin/env bash
set -e

SYMBOL=${1:-"AAPL"}
MODEL=${2:-"chronos"}

echo "Running backtest: symbol=$SYMBOL model=$MODEL"
python -m src.evaluation.backtesting --symbol "$SYMBOL" --model "$MODEL"
