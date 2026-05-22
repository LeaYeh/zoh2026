from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
import wandb


@dataclass
class BacktestResult:
    symbol: str
    n_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    decisions: list[dict] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    decision_fn: Callable[[str, float], dict],
    symbol: str,
    initial_capital: float = 100_000,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Walk-forward backtest. `decision_fn(symbol, price) -> {"action": BUY|SELL|HOLD}`."""
    capital = initial_capital
    position = 0.0
    returns = []
    decisions = []

    prices = df["close"].values
    for i in range(1, len(prices)):
        price = prices[i]
        result = decision_fn(symbol, price)
        action = result.get("action", "HOLD")

        if action == "BUY" and capital > 0:
            shares = capital / (price * (1 + transaction_cost))
            position += shares
            capital = 0.0
        elif action == "SELL" and position > 0:
            capital = position * price * (1 - transaction_cost)
            position = 0.0

        portfolio_value = capital + position * price
        returns.append(portfolio_value)
        decisions.append({"date": str(df.index[i]), "action": action, "price": price, "portfolio": portfolio_value})

    returns_arr = np.array(returns)
    pct_returns = np.diff(returns_arr) / returns_arr[:-1]
    sharpe = float(np.mean(pct_returns) / (np.std(pct_returns) + 1e-9) * np.sqrt(252))
    total_return = float((returns_arr[-1] - initial_capital) / initial_capital)
    drawdown = float(np.max(np.maximum.accumulate(returns_arr) - returns_arr) / np.maximum.accumulate(returns_arr).max())
    wins = sum(1 for d in decisions if d["action"] == "BUY")
    win_rate = wins / max(len(decisions), 1)

    return BacktestResult(
        symbol=symbol,
        n_trades=len([d for d in decisions if d["action"] != "HOLD"]),
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        win_rate=win_rate,
        decisions=decisions,
    )


def log_backtest(result: BacktestResult, run_name: str) -> None:
    wandb.init(project=None, name=run_name, config={"symbol": result.symbol})
    wandb.log({
        "backtest/total_return": result.total_return,
        "backtest/sharpe_ratio": result.sharpe_ratio,
        "backtest/max_drawdown": result.max_drawdown,
        "backtest/win_rate": result.win_rate,
        "backtest/n_trades": result.n_trades,
    })
    wandb.finish()
