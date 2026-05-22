"""Comparative experiments — deterministic + seeded (§4 cross-ticker, §7 reward design).

    uv run python scripts/compare_experiments.py --episodes 300

Two controlled experiments the brief mandates:
  - §7 reward design: same data/agent, only the reward differs — a **basic** reward
    (portfolio-value change only) vs the **full** reward (− cost − slippage + λ·Sharpe).
  - §4 cross-ticker: the AAPL pipeline re-run on **NVDA** (same data mechanism), to
    show the result is a property of the method, not an AAPL quirk.

Writes ``results/analysis/{reward_comparison,cross_ticker}.{json,png}`` and persists a
``data/raw/{ticker}.csv`` fallback for any newly-fetched ticker (offline reproducibility).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from tradedqn.config import load_config
from tradedqn.sdk import TradingSDK

OUT = Path("results/analysis")
METRIC_KEYS = ("total_return", "benchmark_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades")


def _cfg(episodes: int, **env_overrides: float) -> object:
    """Fresh config with the training budget + optional env-reward overrides."""
    cfg = load_config("config/config.yaml")
    cfg.training.episodes = episodes
    for key, value in env_overrides.items():
        setattr(cfg.env, key, value)
    return cfg


def _run(cfg, ticker: str | None = None) -> dict:
    """Prepare → train → greedy backtest on the held-out test split."""
    sdk = TradingSDK(cfg=cfg)
    sdk.prepare_data(ticker=ticker)
    sdk.train()
    result = sdk.backtest("test")
    return {
        "equity_curve": result["equity_curve"],
        "benchmark_curve": result["benchmark_curve"],
        "metrics": {key: result[key] for key in METRIC_KEYS},
    }


def _persist_csv(ticker: str, cfg) -> None:
    """Save the fetched OHLCV as the committed CSV fallback (offline reproducibility)."""
    d = cfg.data
    parquet = Path(d.cache_dir) / f"{ticker}_{d.start}_{d.end}.parquet"
    csv = Path(d.cache_dir) / f"{ticker}.csv"
    if parquet.exists() and not csv.exists():
        pd.read_parquet(parquet).to_csv(csv, index_label="Date")


def _equity_figure(title: str, series: dict[str, list[float]]) -> Figure:
    """Overlay equity curves (one line per labelled run) for a comparison."""
    fig = Figure(figsize=(7.0, 4.0), dpi=160)
    ax = fig.add_subplot(111)
    for label, equity in series.items():
        ax.plot(equity, label=label)
    ax.set_title(title)
    ax.set_xlabel("trading step")
    ax.set_ylabel("portfolio value")
    ax.legend()
    fig.tight_layout()
    return fig


def reward_comparison(episodes: int) -> dict:
    """§7 — basic (ΔV only) vs full (ΔV − cost − slippage + λ·Sharpe) reward on AAPL."""
    full = _run(_cfg(episodes))
    basic = _run(_cfg(episodes, transaction_cost=0.0, slippage=0.0, risk_lambda=0.0))
    fig = _equity_figure(
        "Reward design on AAPL test — basic ΔV vs full risk/cost-adjusted",
        {"Full reward": full["equity_curve"], "Basic reward (ΔV only)": basic["equity_curve"]},
    )
    fig.savefig(OUT / "reward_comparison.png", bbox_inches="tight")
    out = {"full": full["metrics"], "basic": basic["metrics"], "episodes": episodes}
    (OUT / "reward_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def cross_ticker(episodes: int, ticker: str) -> dict:
    """§4 — re-run the AAPL pipeline on another ticker via the same data mechanism."""
    cfg = _cfg(episodes)
    run = _run(cfg, ticker=ticker)
    _persist_csv(ticker, cfg)
    fig = _equity_figure(
        f"{ticker} test — DQN policy vs Buy & Hold ({episodes} episodes)",
        {"DQN policy": run["equity_curve"], "Buy & Hold": run["benchmark_curve"]},
    )
    fig.savefig(OUT / "cross_ticker.png", bbox_inches="tight")
    out = {"ticker": ticker, "metrics": run["metrics"], "episodes": episodes}
    (OUT / "cross_ticker.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--ticker", default="NVDA", help="cross-ticker symbol (SPY or NVDA)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("reward comparison:", json.dumps(reward_comparison(args.episodes), indent=2), flush=True)
    print("cross-ticker:", json.dumps(cross_ticker(args.episodes, args.ticker), indent=2), flush=True)


if __name__ == "__main__":
    main()
