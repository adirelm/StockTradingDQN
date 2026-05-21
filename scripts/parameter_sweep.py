"""One-at-a-time (OAT) sensitivity sweep on the VALIDATION split (§9.1).

Varies one hyperparameter at a time (learning_rate, gamma), holding the rest at
config defaults, and records held-out **validation** metrics — never the test
set, so this analysis can't leak into the reported test result. Uses the cached
AAPL data (no network). Writes results/analysis/sweep.json + sweep.png.

    uv run python scripts/parameter_sweep.py --episodes 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

from tradedqn.config import load_config
from tradedqn.data.client import DataClient
from tradedqn.sdk import TradingSDK

GRID = {"learning_rate": [0.0005, 0.001, 0.005], "gamma": [0.90, 0.95, 0.99]}


def _evaluate(param: str, value: float, episodes: int, client: DataClient) -> dict:
    cfg = load_config("config/config.yaml")
    setattr(cfg.training, param, value)
    sdk = TradingSDK(cfg=cfg, data_client=client)
    sdk.prepare_data()
    sdk.train(episodes)
    val = sdk.backtest("validation")
    return {"value": value, "val_sharpe": val["sharpe_ratio"],
            "val_return": val["total_return"], "trades": val["num_trades"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = DataClient(cache_dir=load_config("config/config.yaml").data.cache_dir)

    results = {}
    for param, values in GRID.items():
        rows = []
        for value in values:
            row = _evaluate(param, value, args.episodes, client)
            rows.append(row)
            print(f"{param}={value}: val_sharpe={row['val_sharpe']:+.3f} "
                  f"val_return={row['val_return']:+.2%} trades={row['trades']}", flush=True)
        results[param] = rows

    (out / "sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(results).savefig(out / "sweep.png", bbox_inches="tight")
    print(json.dumps(results, indent=2))


def _plot(results: dict) -> Figure:
    fig = Figure(figsize=(9.0, 3.4), dpi=160)
    for i, (param, rows) in enumerate(results.items(), start=1):
        ax = fig.add_subplot(1, len(results), i)
        ax.plot([r["value"] for r in rows], [r["val_sharpe"] for r in rows], "o-", color="#1f77b4")
        ax.set_title(f"Validation Sharpe vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("validation Sharpe")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
