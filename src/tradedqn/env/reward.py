"""RewardFunction — the deck's risk/cost-adjusted reward.

``rₜ = ΔVₜ − Cₜ − Sₜ + λ·Sharpeₜ`` (all in fraction-of-initial-capital units).
``ΔV`` is the market PnL on held shares this step; ``C``/``S`` are the step's
trade fees; ``Sharpe`` is the rolling mean/std of recent net step returns.
Components are returned alongside the scalar reward for transparency.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class RewardFunction:
    def __init__(self, risk_lambda: float, sharpe_window: int) -> None:
        self.risk_lambda = float(risk_lambda)
        self.sharpe_window = int(sharpe_window)
        self._returns: deque[float] = deque(maxlen=self.sharpe_window)

    def reset(self) -> None:
        self._returns.clear()

    def _rolling_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        arr = np.asarray(self._returns, dtype=float)
        std = arr.std()
        return 0.0 if std == 0.0 else float(arr.mean() / std)

    def compute(
        self, delta_v: float, cost: float, slippage: float, initial_capital: float
    ) -> tuple[float, dict[str, float]]:
        """Return ``(reward, components)`` for one step."""
        step_return = (delta_v - cost - slippage) / initial_capital
        self._returns.append(step_return)
        sharpe = self._rolling_sharpe()
        reward = step_return + self.risk_lambda * sharpe
        components = {
            "delta_v": delta_v / initial_capital,
            "cost": cost / initial_capital,
            "slippage": slippage / initial_capital,
            "step_return": step_return,
            "sharpe": sharpe,
            "reward": reward,
        }
        return reward, components
