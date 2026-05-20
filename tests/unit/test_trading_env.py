"""Tests for TradingEnvironment (B6 — 30×10 state, step/reset, no look-ahead)."""

import numpy as np
import pandas as pd
import pytest

from tradedqn.config import Config
from tradedqn.env.trading_env import TradingEnvironment

WINDOW = 3


def make_cfg():
    return Config(
        {
            "features": {"window_size": WINDOW},
            "env": {"initial_capital": 1000.0, "transaction_cost": 0.001,
                    "slippage": 0.0005, "risk_lambda": 0.1, "sharpe_window": 5},
            "actions": {"sell": 0, "hold": 1, "buy": 2},
        }
    )


@pytest.fixture
def env():
    n = 8
    feats = pd.DataFrame({f"f{j}": [float(i) for i in range(n)] for j in range(8)})
    prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]  # strictly rising
    return TradingEnvironment(feats, prices, make_cfg())


class TestState:
    def test_reset_shape_and_dtype(self, env):
        state = env.reset()
        assert state.shape == (WINDOW, 10)
        assert state.dtype == np.float32

    def test_portfolio_channels_after_reset(self, env):
        state = env.reset()
        assert np.allclose(state[:, 8], 0.0)   # position: all cash → 0
        assert np.allclose(state[:, 9], 1.0)   # cash_exposure: all cash → 1

    def test_no_look_ahead_last_row_is_current_day(self, env):
        env.reset()  # t = WINDOW-1 = 2
        state = env.reset()
        assert state[-1, 0] == pytest.approx(2.0)   # feature row index 2 = current day
        assert state[-1, 0] != 3.0                  # never the not-yet-known next row


class TestStep:
    def test_buy_then_price_up_positive_reward(self, env):
        env.reset()
        _, reward, _, info = env.step(2)  # Buy at 12.0, next price 13.0
        assert reward > 0.0
        assert info["action"] == "buy"

    def test_hold_incurs_no_cost(self, env):
        env.reset()
        _, _, _, info = env.step(1)
        assert info["cost"] == 0.0 and info["slippage"] == 0.0

    def test_episode_terminates_at_end(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(1)
            steps += 1
            assert steps < 100
        assert done


class TestConstruction:
    def test_length_mismatch_raises(self):
        feats = pd.DataFrame({f"f{j}": [1.0, 2.0, 3.0, 4.0] for j in range(8)})
        with pytest.raises(ValueError, match="same length"):
            TradingEnvironment(feats, [1.0, 2.0, 3.0], make_cfg())

    def test_too_few_rows_raises(self):
        feats = pd.DataFrame({f"f{j}": [1.0, 2.0] for j in range(8)})
        with pytest.raises(ValueError, match="not enough rows"):
            TradingEnvironment(feats, [1.0, 2.0], make_cfg())
