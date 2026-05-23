"""Reward-decomposition waterfall (§9.3) — REAL aggregated test-backtest components.

    uv run python scripts/reward_waterfall.py [--episodes 300]

Trains the headline policy (seeded → reproduces the −17.5 % test run), then replays
it greedily over the held-out AAPL test slice and sums the env's per-step reward
components (``rₜ = ΔVₜ − Cₜ − Sₜ + λ·Sharpeₜ``, fraction-of-initial-capital units)
into a waterfall: ΔV → −C → −S → +λ·Sharpe → net reward. The four bars sum exactly
to Σ reward (definitional), so nothing is fabricated. Offline + deterministic off
the committed cache. Writes ``results/analysis/reward_waterfall.{json,png}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

from tradedqn.sdk import TradingSDK

OUT = Path("results/analysis")
PARTS = ("delta_v", "cost", "slippage", "sharpe", "reward")


def rollout_components(sdk: TradingSDK, split: str = "test") -> dict:
    """Greedy rollout over ``split``; sum the env info's reward components + final value."""
    env = sdk._env(split)
    totals = dict.fromkeys(PARTS, 0.0)
    state, done, final_value = env.reset(), False, env.initial_capital
    while not done:
        state, _, done, info = env.step(sdk.agent.act(state, greedy=True))
        for key in PARTS:
            totals[key] += float(info[key])
        final_value = float(info["value"])
    totals["total_return"] = final_value / env.initial_capital - 1.0
    totals["risk_lambda"] = float(sdk.cfg.env.risk_lambda)
    return totals


def waterfall_figure(t: dict) -> Figure:
    """Bar-waterfall ΔV → −C → −S → +λ·Sharpe → net reward (summed reward units).

    The capital-PnL terms (ΔV/cost/slippage, fraction-of-capital) and the unitless
    summed rolling-Sharpe term share the reward formula but not a scale — each bar
    is value-labelled so the small terms stay legible next to the dominant Sharpe.
    """
    lam = t["risk_lambda"]
    steps = [("ΔV (market PnL)", t["delta_v"]), ("− cost", -t["cost"]),
             ("− slippage", -t["slippage"]), ("+ λ·Sharpe", lam * t["sharpe"])]
    fig = Figure(figsize=(7.0, 4.0), dpi=160)
    ax = fig.add_subplot(111)
    running = 0.0
    for label, delta in steps:
        ax.bar(label, delta, bottom=running, color="#2ca02c" if delta >= 0 else "#d62728")
        ax.text(label, running + delta, f"{delta:+.3f}", ha="center",
                va="top" if delta < 0 else "bottom", fontsize=8)
        running += delta
    ax.bar("net reward", running, color="#1f77b4")
    ax.text("net reward", running, f"{running:+.3f}", ha="center", va="top", fontsize=8)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_title("Reward decomposition — AAPL test (summed; the λ·Sharpe term dominates)")
    ax.set_ylabel("summed reward contribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sdk = TradingSDK()
    sdk.prepare_data()
    sdk.train(args.episodes)
    totals = rollout_components(sdk, "test")
    waterfall_figure(totals).savefig(OUT / "reward_waterfall.png", bbox_inches="tight")
    (OUT / "reward_waterfall.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
