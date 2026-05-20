"""Tests for Portfolio (B7/B10 — all-in/all-out accounting with fees)."""

import pytest

from tradedqn.env.portfolio import Portfolio


@pytest.fixture
def pf():
    return Portfolio(initial_capital=1000.0, transaction_cost=0.001, slippage=0.0005)


class TestBuy:
    def test_buy_spends_all_cash_minus_fees(self, pf):
        trade = pf.buy(price=10.0)
        assert pf.cash == 0.0
        assert pf.shares > 0.0
        assert trade["cost"] == pytest.approx(1.0)        # 1000 * 0.001
        assert trade["slippage"] == pytest.approx(0.5)    # 1000 * 0.0005

    def test_buy_noop_when_no_cash(self, pf):
        pf.buy(10.0)
        again = pf.buy(10.0)
        assert again["traded"] == 0.0


class TestSell:
    def test_sell_liquidates_all_holdings(self, pf):
        pf.buy(10.0)
        pf.sell(12.0)
        assert pf.shares == 0.0
        assert pf.cash > 0.0

    def test_sell_noop_when_flat(self, pf):
        assert pf.sell(10.0)["traded"] == 0.0


class TestValuation:
    def test_value_is_cash_plus_marked_shares(self, pf):
        pf.buy(10.0)
        assert pf.value(10.0) == pytest.approx(pf.shares * 10.0)

    def test_position_and_cash_exposure_sum_to_one(self, pf):
        pf.buy(10.0)
        assert pf.position(11.0) + pf.cash_exposure(11.0) == pytest.approx(1.0)

    def test_all_cash_means_zero_position(self, pf):
        assert pf.position(10.0) == 0.0
        assert pf.cash_exposure(10.0) == pytest.approx(1.0)

    def test_reset_restores_initial(self, pf):
        pf.buy(10.0)
        pf.reset()
        assert (pf.cash, pf.shares) == (1000.0, 0.0)
