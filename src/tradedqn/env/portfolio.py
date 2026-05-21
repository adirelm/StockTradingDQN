"""Portfolio — all-in / all-out cash+shares accounting with trade fees.

Buy invests *all* cash; Sell liquidates *all* holdings; Hold does nothing
(the deck's simple "have stock or not" model). Each trade returns its
transaction-cost and slippage so the reward function can charge them.
"""

from __future__ import annotations

Trade = dict[str, float]
_NO_TRADE: Trade = {"cost": 0.0, "slippage": 0.0, "traded": 0.0}


class Portfolio:
    """All-in / all-out holdings: tracks cash + shares and charges fees per trade."""

    def __init__(self, initial_capital: float, transaction_cost: float, slippage: float) -> None:
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.cash = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0

    def reset(self) -> None:
        """Restore the opening balance (all cash, no shares)."""
        self.cash = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0

    def _fees(self, notional: float) -> Trade:
        """Transaction cost + slippage for a trade of the given notional size."""
        return {
            "cost": notional * self.transaction_cost,
            "slippage": notional * self.slippage,
            "traded": notional,
        }

    def buy(self, price: float) -> Trade:
        """Invest all cash at ``price`` (minus fees). No-op if flat on cash."""
        if self.cash <= 0.0 or price <= 0.0:
            return dict(_NO_TRADE)
        fees = self._fees(self.cash)
        invested = self.cash - fees["cost"] - fees["slippage"]
        self.shares += invested / price
        self.cash = 0.0
        self.entry_price = price
        return fees

    def sell(self, price: float) -> Trade:
        """Liquidate all holdings at ``price`` (minus fees). No-op if holding nothing."""
        if self.shares <= 0.0 or price <= 0.0:
            return dict(_NO_TRADE)
        notional = self.shares * price
        fees = self._fees(notional)
        self.cash += notional - fees["cost"] - fees["slippage"]
        self.shares = 0.0
        self.entry_price = 0.0
        return fees

    def hold(self, price: float | None = None) -> Trade:
        """Do nothing this step; return a zero-cost trade record."""
        return dict(_NO_TRADE)

    def value(self, price: float) -> float:
        """Mark-to-market portfolio value = cash + shares · price."""
        return self.cash + self.shares * price

    def position(self, price: float) -> float:
        """Fraction of portfolio value held in the asset (∈ [0, 1])."""
        total = self.value(price)
        return 0.0 if total <= 0.0 else (self.shares * price) / total

    def unrealized_pnl(self, price: float) -> float:
        """Open-position P&L as a fraction of initial capital (0 when flat)."""
        if self.shares <= 0.0:
            return 0.0
        return (price - self.entry_price) * self.shares / self.initial_capital
