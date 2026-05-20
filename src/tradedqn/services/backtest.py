"""BacktestService — replay the greedy policy over a held-out slice.

Reports the strategy equity curve against a Buy & Hold benchmark plus the
deck's metrics (total return, Sharpe, max drawdown, win rate, num trades).
This is the project's "prove the policy works" evidence. Past ≠ future.
"""

from __future__ import annotations

from tradedqn.services import metrics


class BacktestService:
    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    def run(self) -> dict:
        """Greedy rollout over the env; return curves + metrics."""
        state = self.env.reset()
        initial = self.env.portfolio.initial_capital
        equity, prices = [initial], []
        trades = wins = round_trips = 0
        entry_equity = None
        done = False
        while not done:
            action = self.agent.act(state, greedy=True)
            state, _, done, info = self.env.step(action)
            equity.append(info["value"])
            prices.append(info["price"])
            if info["traded"] > 0:
                trades += 1
                if info["action"] == "buy":
                    entry_equity = info["value"]
                elif info["action"] == "sell" and entry_equity is not None:
                    round_trips += 1
                    wins += int(info["value"] > entry_equity)
                    entry_equity = None
        return self._summary(equity, prices, trades, wins, round_trips, initial)

    @staticmethod
    def _benchmark(prices, initial: float) -> list[float]:
        base = prices[0]
        return [initial * price / base for price in prices]

    def _summary(self, equity, prices, trades, wins, round_trips, initial) -> dict:
        benchmark = self._benchmark(prices, initial) if prices else [initial]
        return {
            "equity_curve": equity,
            "benchmark_curve": [initial, *benchmark],
            "total_return": metrics.total_return(equity),
            "benchmark_return": metrics.total_return([initial, *benchmark]),
            "sharpe_ratio": metrics.sharpe_ratio(equity),
            "max_drawdown": metrics.max_drawdown(equity),
            "win_rate": (wins / round_trips) if round_trips else 0.0,
            "num_trades": trades,
        }
