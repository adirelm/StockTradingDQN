"""Integration test for BacktestService (B17 — curves + metrics vs Buy&Hold)."""

import math

from tradedqn.services.backtest import BacktestService


class ScriptedAgent:
    """Plays a fixed action sequence (then holds) — to exercise trade counting."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, state, greedy=True):
        action = self.actions[self.i] if self.i < len(self.actions) else 1  # hold
        self.i += 1
        return action


class TestBacktest:
    def test_run_returns_full_summary(self, toy_env, dqn_agent):
        result = BacktestService(toy_env, dqn_agent).run()
        expected = {
            "equity_curve", "benchmark_curve", "total_return", "benchmark_return",
            "sharpe_ratio", "max_drawdown", "win_rate", "num_trades",
        }
        assert expected <= result.keys()

    def test_curves_equal_length_and_start_at_capital(self, toy_env, dqn_agent):
        result = BacktestService(toy_env, dqn_agent).run()
        assert len(result["equity_curve"]) == len(result["benchmark_curve"])
        assert result["equity_curve"][0] == toy_env.portfolio.initial_capital
        assert result["benchmark_curve"][0] == toy_env.portfolio.initial_capital

    def test_metrics_are_finite_and_sane(self, toy_env, dqn_agent):
        result = BacktestService(toy_env, dqn_agent).run()
        assert math.isfinite(result["total_return"])
        assert math.isfinite(result["sharpe_ratio"])
        assert 0.0 <= result["max_drawdown"] <= 1.0
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["num_trades"] >= 0

    def test_greedy_is_deterministic_across_runs(self, toy_env, dqn_agent):
        a = BacktestService(toy_env, dqn_agent).run()
        b = BacktestService(toy_env, dqn_agent).run()
        assert a["equity_curve"] == b["equity_curve"]  # greedy → reproducible

    def test_winning_round_trip_counts(self, toy_env):
        # Buy at the start, hold, then Sell — on a strictly rising series this wins
        agent = ScriptedAgent([2, 1, 1, 1, 1, 0])  # buy, holds, sell
        result = BacktestService(toy_env, agent).run()
        assert result["num_trades"] == 2          # one buy + one sell
        assert result["win_rate"] == 1.0          # sold higher than bought
        assert result["total_return"] > 0.0
