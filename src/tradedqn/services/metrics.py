"""Backtest metrics — pure functions over an equity-value series."""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def total_return(equity) -> float:
    """Final / initial − 1."""
    equity = np.asarray(equity, dtype=float)
    return float(equity[-1] / equity[0] - 1.0)


def sharpe_ratio(equity, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe of step returns; 0 when there is no variation."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return 0.0
    step_returns = np.diff(equity) / equity[:-1]
    std = step_returns.std()
    if std == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * step_returns.mean() / std)


def max_drawdown(equity) -> float:
    """Largest peak-to-trough drop as a positive fraction of the running peak."""
    equity = np.asarray(equity, dtype=float)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (running_peak - equity) / running_peak
    return float(drawdowns.max())
