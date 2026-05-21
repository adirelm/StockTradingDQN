"""Generate README results: train on the configured ticker, backtest, save artifacts.

    uv run python scripts/generate_results.py --episodes 300

Writes to ``results/analysis/``: training_reward.png, backtest_equity.png,
backtest_metrics.json. Hits Yahoo Finance once (gatekept + cached) on first run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradedqn.gui.charts import equity_figure, training_figure
from tradedqn.sdk import TradingSDK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None, help="override training episodes")
    parser.add_argument("--out", default="results/analysis", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sdk = TradingSDK()
    splits = sdk.prepare_data()
    print(f"splits: {splits}", flush=True)
    episodes = args.episodes if args.episodes is not None else sdk.cfg.training.episodes
    history = []
    for i in range(episodes):
        record = sdk.train(1)[0]
        record["episode"] = i
        history.append(record)
        loss = record["mean_loss"]
        loss_str = f"{loss:.4f}" if loss is not None else "warmup"
        print(
            f"ep {i + 1:>3}/{episodes}  reward={record['reward']:+.4f}  "
            f"final={record['final_value']:.1f}  eps={record['epsilon']:.3f}  loss={loss_str}",
            flush=True,
        )
    result = sdk.backtest("test")
    recommendation = sdk.recommend("test")

    training_figure(history).savefig(out / "training_reward.png", bbox_inches="tight")
    equity_figure(result["equity_curve"], result["benchmark_curve"]).savefig(
        out / "backtest_equity.png", bbox_inches="tight"
    )

    summary = {key: value for key, value in result.items() if "curve" not in key}
    summary.update(
        ticker=sdk.cfg.data.ticker,
        period=f"{sdk.cfg.data.start}..{sdk.cfg.data.end}",
        episodes=len(history),
        splits=splits,
        recommendation=recommendation["action"],
    )
    (out / "backtest_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
