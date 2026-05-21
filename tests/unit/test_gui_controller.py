"""Tests for GuiController (B22 — GUI logic with no Tk dependency)."""

import pytest
from matplotlib.figure import Figure

from tradedqn.gui.controller import GuiController


class FakeSDK:
    def __init__(self, fail=False):
        self._fail = fail

    def prepare_data(self):
        if self._fail:
            raise RuntimeError("boom")
        return {"train": 100, "validation": 20, "test": 20}

    def train(self, episodes=None):
        return [{"episode": 0, "epsilon": 0.4, "reward": 1.0},
                {"episode": 1, "epsilon": 0.2, "reward": 2.0}]

    def backtest(self, split="test"):
        return {"total_return": 0.12, "benchmark_return": 0.08, "sharpe_ratio": 1.3,
                "max_drawdown": 0.05, "num_trades": 4,
                "equity_curve": [1000, 1100], "benchmark_curve": [1000, 1080]}

    def recommend(self, split="test"):
        return {"action": "buy", "action_index": 2, "q_values": [0.1, 0.2, 0.3]}


class TestController:
    def test_prepare_status_has_sizes(self):
        assert "train" in GuiController(FakeSDK()).prepare()

    def test_train_returns_status_and_figure(self):
        status, fig = GuiController(FakeSDK()).train()
        assert isinstance(fig, Figure) and "ε=0.200" in status

    def test_backtest_returns_figure_and_caches(self):
        ctrl = GuiController(FakeSDK())
        status, fig = ctrl.backtest()
        assert isinstance(fig, Figure)
        assert "12.00%" in status
        assert ctrl.last_backtest["num_trades"] == 4

    def test_recommend_status_has_action(self):
        assert "BUY" in GuiController(FakeSDK()).recommend()

    def test_sdk_error_propagates(self):
        with pytest.raises(RuntimeError, match="boom"):
            GuiController(FakeSDK(fail=True)).prepare()
