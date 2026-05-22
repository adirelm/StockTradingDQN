"""Shared presentation helpers for the terminal and GUI (no duplication — §3.2/§4.2).

Both UIs render the recommendation and backtest summary identically; the format
lives here once so the wording stays consistent across interfaces (§10).
"""

from __future__ import annotations


def recommendation_line(rec: dict) -> str:
    """One-line Buy/Hold/Sell recommendation: Q-vector, confidence, top feature drivers."""
    quotes = ", ".join(f"{value:.3f}" for value in rec["q_values"])
    line = f"Recommended action: {rec['action'].upper()}  (Q = [{quotes}])"
    if rec.get("confidence") is not None:
        line += f"  ·  confidence {rec['confidence']:.0%}"
    if rec.get("top_features"):
        line += f"  ·  drivers: {', '.join(rec['top_features'])}"
    return line


def backtest_line(result: dict) -> str:
    """One-line backtest summary (return vs benchmark + risk/cost metrics)."""
    return (
        f"total_return={result['total_return']:.2%}  benchmark={result['benchmark_return']:.2%}  "
        f"Sharpe={result['sharpe_ratio']:.2f}  max_drawdown={result['max_drawdown']:.2%}  "
        f"win_rate={result['win_rate']:.2%}  trades={result['num_trades']}"
    )
