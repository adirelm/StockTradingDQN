"""Excellence ablations — deterministic, seeded (§6 Double-DQN, §9 seed robustness).

    uv run python scripts/ablations.py --episodes 300

Two studies the brief lists as excellence directions:
  - §6 Double-DQN vs vanilla DQN — same trunk, only the Bellman *target* differs
    (vanilla: target net max; double: online net selects, target net evaluates —
    reduces value over-estimation).
  - §9 seed robustness — the headline run repeated across several seeds, reported
    as mean ± std, so the single-seed result isn't read as a fluke.

Writes ``results/analysis/{double_q,seed_variance}.{json,png}``.
Uses the committed cached data, so it is offline + bit-for-bit reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from matplotlib.figure import Figure

from tradedqn.config import load_config
from tradedqn.sdk import TradingSDK

OUT = Path("results/analysis")
KEYS = ("total_return", "benchmark_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades")


def _run(episodes: int, *, seed: int | None = None, double_q: bool = False) -> dict:
    """Prepare → train → greedy test backtest; return the metrics + equity curve."""
    cfg = load_config("config/config.yaml")
    cfg.training.episodes = episodes
    cfg.training.double_q = double_q
    if seed is not None:
        cfg.seed = seed
    sdk = TradingSDK(cfg=cfg)
    sdk.prepare_data()
    sdk.train()
    bt = sdk.backtest("test")
    return {"metrics": {k: bt[k] for k in KEYS}, "equity": bt["equity_curve"]}


def double_q_comparison(episodes: int) -> dict:
    """§6 — vanilla DQN vs Double DQN on the held-out AAPL test slice."""
    vanilla = _run(episodes, double_q=False)
    double = _run(episodes, double_q=True)
    fig = Figure(figsize=(7.0, 4.0), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(vanilla["equity"], label="Vanilla DQN")
    ax.plot(double["equity"], label="Double DQN")
    ax.set_title(f"Vanilla vs Double DQN — AAPL test ({episodes} episodes)")
    ax.set_xlabel("trading step")
    ax.set_ylabel("portfolio value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "double_q.png", bbox_inches="tight")
    out = {"vanilla": vanilla["metrics"], "double": double["metrics"], "episodes": episodes}
    (OUT / "double_q.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _plot_seed_variance(out: dict) -> None:
    """Bar each seed's test return against the mean and Buy & Hold reference lines."""
    seeds = out["seeds"]
    returns = [out["per_seed"][str(s)]["total_return"] * 100 for s in seeds]
    mean = out["summary"]["total_return"]["mean"] * 100
    bench = out["per_seed"][str(seeds[0])]["benchmark_return"] * 100
    fig = Figure(figsize=(7.0, 4.0), dpi=160)
    ax = fig.add_subplot(111)
    ax.bar([str(s) for s in seeds], returns, color="#4c72b0")
    ax.axhline(mean, color="#dd8452", ls="--", label=f"mean {mean:.1f}%")
    ax.axhline(bench, color="#555555", ls=":", label=f"Buy & Hold {bench:.1f}%")
    ax.set_title(f"Seed robustness — AAPL test return ({out['episodes']} episodes)")
    ax.set_xlabel("seed")
    ax.set_ylabel("total return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "seed_variance.png", bbox_inches="tight")


def seed_variance(episodes: int, seeds: list[int]) -> dict:
    """§9 — repeat the headline run across seeds; report mean ± std (robustness)."""
    runs = {str(seed): _run(episodes, seed=seed)["metrics"] for seed in seeds}
    summary = {}
    for metric in ("total_return", "sharpe_ratio"):
        values = [runs[str(s)][metric] for s in seeds]
        summary[metric] = {"mean": statistics.mean(values), "std": statistics.pstdev(values),
                           "min": min(values), "max": max(values)}
    out = {"seeds": seeds, "episodes": episodes, "per_seed": runs, "summary": summary}
    (OUT / "seed_variance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _plot_seed_variance(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 13, 100])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("double_q:", json.dumps(double_q_comparison(args.episodes), indent=2), flush=True)
    print("seed_variance:", json.dumps(seed_variance(args.episodes, args.seeds), indent=2), flush=True)


if __name__ == "__main__":
    main()
