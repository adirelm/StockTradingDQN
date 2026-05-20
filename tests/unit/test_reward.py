"""Tests for RewardFunction (B10 — r = ΔV − C − S + λ·Sharpe)."""

import pytest

from tradedqn.env.reward import RewardFunction


class TestComposition:
    def test_first_step_has_zero_sharpe(self):
        reward, comp = RewardFunction(risk_lambda=0.1, sharpe_window=5).compute(
            delta_v=50.0, cost=1.0, slippage=0.5, initial_capital=1000.0
        )
        assert comp["sharpe"] == 0.0
        assert reward == pytest.approx(comp["step_return"])  # λ·0 term vanishes
        assert comp["step_return"] == pytest.approx((50.0 - 1.0 - 0.5) / 1000.0)

    def test_components_are_fraction_units(self):
        _, comp = RewardFunction(0.0, 5).compute(20.0, 2.0, 1.0, 1000.0)
        assert comp["delta_v"] == pytest.approx(0.02)
        assert comp["cost"] == pytest.approx(0.002)
        assert comp["slippage"] == pytest.approx(0.001)

    def test_lambda_zero_removes_sharpe_term(self):
        rf = RewardFunction(risk_lambda=0.0, sharpe_window=5)
        for _ in range(4):
            reward, comp = rf.compute(30.0, 0.0, 0.0, 1000.0)
        assert reward == pytest.approx(comp["step_return"])

    def test_hold_with_no_fees_is_pure_return(self):
        _, comp = RewardFunction(0.0, 5).compute(delta_v=10.0, cost=0.0, slippage=0.0, initial_capital=1000.0)
        assert comp["step_return"] == pytest.approx(0.01)

    def test_sharpe_contributes_after_two_steps(self):
        rf = RewardFunction(risk_lambda=1.0, sharpe_window=5)
        rf.compute(10.0, 0.0, 0.0, 1000.0)
        reward, comp = rf.compute(20.0, 0.0, 0.0, 1000.0)
        assert comp["sharpe"] != 0.0
        assert reward == pytest.approx(comp["step_return"] + comp["sharpe"])

    def test_reset_clears_history(self):
        rf = RewardFunction(1.0, 5)
        rf.compute(10.0, 0.0, 0.0, 1000.0)
        rf.reset()
        _, comp = rf.compute(10.0, 0.0, 0.0, 1000.0)
        assert comp["sharpe"] == 0.0
