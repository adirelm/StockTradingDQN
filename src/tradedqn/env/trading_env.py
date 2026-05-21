"""TradingEnvironment — Gym-style env producing the 30×10 state.

Channels 1–8 are the per-day market features; channels 9–10 are the agent's
current ``position`` and ``cash_exposure`` broadcast across the window. The
state at decision day ``t`` uses only feature rows up to and including ``t`` and
executes at ``prices[t]`` — the next day's price is the *outcome*, never part of
the observed state (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradedqn.env.portfolio import Portfolio
from tradedqn.env.reward import RewardFunction


def assemble_state(market_window: np.ndarray, position: float, cash_exposure: float) -> np.ndarray:
    """Stack a market window (window, 8) with the 2 broadcast portfolio channels."""
    portfolio = np.full((market_window.shape[0], 2), [position, cash_exposure], dtype=np.float32)
    return np.hstack([market_window, portfolio]).astype(np.float32)


class TradingEnvironment:
    def __init__(self, features: pd.DataFrame, prices, cfg) -> None:
        self._features = features.to_numpy(dtype=np.float32)
        self._prices = np.asarray(prices, dtype=np.float64)
        if len(self._features) != len(self._prices):
            raise ValueError("features and prices must have the same length")
        self.window = int(cfg.features.window_size)
        if len(self._prices) <= self.window:
            raise ValueError("not enough rows for one window plus a step")
        self.initial_capital = float(cfg.env.initial_capital)
        self.portfolio = Portfolio(self.initial_capital, cfg.env.transaction_cost, cfg.env.slippage)
        self.reward_fn = RewardFunction(cfg.env.risk_lambda, cfg.env.sharpe_window)
        self._actions = {cfg.actions.sell: "sell", cfg.actions.hold: "hold", cfg.actions.buy: "buy"}
        self._t = self.window - 1

    def reset(self) -> np.ndarray:
        self.portfolio.reset()
        self.reward_fn.reset()
        self._t = self.window - 1
        return self._state()

    def _state(self) -> np.ndarray:
        market = self._features[self._t - self.window + 1 : self._t + 1]  # (window, 8)
        price = self._prices[self._t]
        return assemble_state(market, self.portfolio.position(price), self.portfolio.cash_exposure(price))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        price_now = self._prices[self._t]
        trade = getattr(self.portfolio, self._actions[action])(price_now)
        shares_after = self.portfolio.shares
        self._t += 1
        price_next = self._prices[self._t]
        delta_v = shares_after * (price_next - price_now)
        reward, components = self.reward_fn.compute(
            delta_v, trade["cost"], trade["slippage"], self.initial_capital
        )
        done = self._t >= len(self._prices) - 1
        info = {
            "value": self.portfolio.value(price_next),
            "action": self._actions[action],
            "price": float(price_next),
            "traded": trade["traded"],
            **components,
        }
        return self._state(), reward, done, info
