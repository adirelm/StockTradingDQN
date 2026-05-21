"""Tests for GuiController (B22 — GUI logic with no Tk dependency)."""

import pytest
from matplotlib.figure import Figure

from tradedqn.gui.controller import GuiController


class FakeSDK:
    def __init__(self, fail=False):
        self._fail = fail

    def prepare_data(self, ticker=None, start=None, end=None):
        if self._fail:
            raise RuntimeError("boom")
        self.prepare_args = (ticker, start, end)
        return {"train": 100, "validation": 20, "test": 20}

    def train(self, episodes=None, on_episode=None):
        self.train_episodes = episodes
        history = [{"episode": 0, "epsilon": 0.4, "reward": 1.0},
                   {"episode": 1, "epsilon": 0.2, "reward": 2.0}]
        if on_episode is not None:
            for record in history:
                on_episode(record)
        return history

    def backtest(self, split="test"):
        return {"total_return": 0.12, "benchmark_return": 0.08, "sharpe_ratio": 1.3,
                "max_drawdown": 0.05, "win_rate": 0.6, "num_trades": 4,
                "equity_curve": [1000, 1100], "benchmark_curve": [1000, 1080],
                "price_curve": [100, 110], "trade_markers": [
                    {"step": 0, "price": 100, "action": "buy"}]}

    def recommend(self, split="test"):
        return {"action": "buy", "action_index": 2, "q_values": [0.1, 0.2, 0.3]}

    def compare(self, episodes=None, on_episode=None):
        hist = {"Dueling DQN": [{"episode": 0, "epsilon": 0.3, "reward": 2.0}],
                "Plain DQN": [{"episode": 0, "epsilon": 0.3, "reward": 1.0}]}
        if on_episode is not None:
            for name, records in hist.items():
                for record in records:
                    on_episode(name, record)
        return hist


class TestController:
    def test_prepare_status_has_sizes(self):
        assert "train" in GuiController(FakeSDK()).prepare()

    def test_prepare_forwards_ticker_and_dates(self):
        sdk = FakeSDK()
        GuiController(sdk).prepare(ticker="MSFT", start="2019-01-01", end="2020-01-01")
        assert sdk.prepare_args == ("MSFT", "2019-01-01", "2020-01-01")

    def test_train_returns_status_and_figure(self):
        status, fig = GuiController(FakeSDK()).train()
        assert isinstance(fig, Figure) and "ε=0.200" in status

    def test_train_forwards_episode_count(self):
        sdk = FakeSDK()
        GuiController(sdk).train(episodes=7)
        assert sdk.train_episodes == 7

    def test_train_reports_live_progress(self):
        seen: list[tuple[str, int]] = []
        GuiController(FakeSDK()).train(
            episodes=2, on_progress=lambda line, hist: seen.append((line, len(hist)))
        )
        assert seen[-1][1] == 2  # history grows live for the animating chart
        assert "Episode 2" in seen[-1][0] and "ε=0.200" in seen[-1][0]

    def test_backtest_returns_figure_and_caches(self):
        ctrl = GuiController(FakeSDK())
        status, fig = ctrl.backtest()
        assert isinstance(fig, Figure)
        assert "12.00%" in status
        assert ctrl.last_backtest["num_trades"] == 4

    def test_recommend_returns_status_and_qvalue_figure(self):
        status, fig = GuiController(FakeSDK()).recommend()
        assert isinstance(fig, Figure) and "BUY" in status

    def test_compare_returns_status_and_figure(self):
        status, fig = GuiController(FakeSDK()).compare(episodes=1)
        assert isinstance(fig, Figure) and "Dueling DQN" in status and "Plain DQN" in status

    def test_compare_streams_progress_per_arch(self):
        seen: list[str] = []
        GuiController(FakeSDK()).compare(episodes=1, on_progress=seen.append)
        assert any("Dueling DQN" in line for line in seen)

    def test_sdk_error_propagates(self):
        with pytest.raises(RuntimeError, match="boom"):
            GuiController(FakeSDK(fail=True)).prepare()
