"""BacktestService — replay the greedy policy over a held-out slice.

Reports the strategy equity curve against a Buy & Hold benchmark plus the
deck's metrics (total return, Sharpe, max drawdown, win rate, num trades).
This is the project's "prove the policy works" evidence. Past ≠ future.
Shares the rollout skeleton with training via :class:`RolloutService`.
"""

from __future__ import annotations

from tradedqn.services import metrics
from tradedqn.services.rollout import RolloutService


class BacktestService(RolloutService):
    """Replays the greedy policy over a split → equity curve, trade markers, metrics."""

    def run(self) -> dict:
        """Greedy rollout over the env; return curves + trade markers + metrics."""
        initial = self.env.portfolio.initial_capital
        acc = {"equity": [initial], "prices": [], "markers": [],
               "trades": 0, "wins": 0, "trips": 0, "entry": None}

        def on_step(state, action, reward, next_state, done, info):
            """Accumulate equity, price, and any trade marker for one step."""
            acc["equity"].append(info["value"])
            acc["prices"].append(info["price"])
            if info["traded"] > 0:
                acc["trades"] += 1
                acc["markers"].append(
                    {"step": len(acc["prices"]) - 1, "price": info["price"], "action": info["action"]}
                )
                if info["action"] == "buy":
                    acc["entry"] = info["value"]
                elif info["action"] == "sell" and acc["entry"] is not None:
                    acc["trips"] += 1
                    acc["wins"] += int(info["value"] > acc["entry"])
                    acc["entry"] = None

        self._rollout(greedy=True, on_step=on_step)
        return self._summary(acc, initial)

    @staticmethod
    def _benchmark(prices, initial: float) -> list[float]:
        """Buy & Hold equity path scaled to the initial capital."""
        base = prices[0]
        return [initial * price / base for price in prices]

    def _summary(self, acc, initial) -> dict:
        """Assemble curves + trade markers + metrics from the accumulated stats."""
        equity, prices = acc["equity"], acc["prices"]
        benchmark = self._benchmark(prices, initial) if prices else [initial]
        return {
            "equity_curve": equity,
            "benchmark_curve": [initial, *benchmark],
            "price_curve": prices,
            "trade_markers": acc["markers"],
            "total_return": metrics.total_return(equity),
            "benchmark_return": metrics.total_return([initial, *benchmark]),
            "sharpe_ratio": metrics.sharpe_ratio(equity),
            "max_drawdown": metrics.max_drawdown(equity),
            "win_rate": (acc["wins"] / acc["trips"]) if acc["trips"] else 0.0,
            "num_trades": acc["trades"],
        }
