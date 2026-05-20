"""Tests for backtest metrics (B17 — total return, Sharpe, max drawdown)."""

import pytest

from tradedqn.services import metrics


class TestTotalReturn:
    def test_simple_gain(self):
        assert metrics.total_return([100.0, 110.0]) == pytest.approx(0.10)

    def test_loss(self):
        assert metrics.total_return([100.0, 80.0]) == pytest.approx(-0.20)


class TestMaxDrawdown:
    def test_peak_to_trough(self):
        assert metrics.max_drawdown([100.0, 120.0, 90.0, 150.0]) == pytest.approx(0.25)

    def test_monotonic_up_has_no_drawdown(self):
        assert metrics.max_drawdown([100.0, 110.0, 130.0]) == pytest.approx(0.0)


class TestSharpe:
    def test_zero_when_returns_constant(self):
        # constant +100% each step → std of returns is 0 → guard returns 0
        assert metrics.sharpe_ratio([1.0, 2.0, 4.0, 8.0]) == 0.0

    def test_single_point_is_zero(self):
        assert metrics.sharpe_ratio([100.0]) == 0.0

    def test_positive_for_upward_varied_series(self):
        assert metrics.sharpe_ratio([100.0, 101.0, 100.5, 103.0, 104.0]) > 0.0
