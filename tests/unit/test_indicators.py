"""Tests for the technical-indicator math (B3 — pure functions on toy series)."""

import pandas as pd
import pytest

from tradedqn.features import indicators as ind


@pytest.fixture
def rising():
    return pd.Series([float(i) for i in range(1, 21)])  # 1..20, strictly increasing


@pytest.fixture
def falling():
    return pd.Series([float(i) for i in range(20, 0, -1)])  # 20..1, strictly decreasing


class TestMovingAverages:
    def test_sma_matches_hand_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert ind.sma(s, 2).tolist()[1:] == [1.5, 2.5, 3.5]

    def test_ema_first_value_equals_seed(self):
        s = pd.Series([10.0, 20.0, 30.0])
        assert ind.ema(s, 2).iloc[0] == 10.0  # adjust=False seeds on first point

    def test_ratio_to_ma_zero_when_flat(self):
        s = pd.Series([5.0] * 10)
        assert ind.ratio_to_ma(s, 3).dropna().abs().max() == pytest.approx(0.0)


class TestRSI:
    def test_all_rising_rsi_is_100(self, rising):
        assert ind.rsi(rising, 5).dropna().iloc[-1] == pytest.approx(100.0)

    def test_all_falling_rsi_is_0(self, falling):
        assert ind.rsi(falling, 5).dropna().iloc[-1] == pytest.approx(0.0)

    def test_rsi_bounded(self):
        s = pd.Series([1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13], dtype=float)
        vals = ind.rsi(s, 4).dropna()
        assert vals.between(0.0, 100.0).all()


class TestMacdAndOthers:
    def test_macd_positive_when_trending_up(self, rising):
        assert ind.macd(rising, 3, 6).dropna().iloc[-1] > 0

    def test_macd_negative_when_trending_down(self, falling):
        assert ind.macd(falling, 3, 6).dropna().iloc[-1] < 0

    def test_high_low_range_is_fraction_of_close(self):
        h, low, c = pd.Series([11.0]), pd.Series([9.0]), pd.Series([10.0])
        assert ind.high_low_range(h, low, c).iloc[0] == pytest.approx(0.2)

    def test_normalized_price_in_unit_range(self):
        s = pd.Series([float(i) for i in range(1, 11)])
        vals = ind.normalized_price(s, 3).dropna()
        assert vals.between(0.0, 1.0).all()

    def test_volatility_zero_for_constant_returns(self):
        s = pd.Series([2.0, 4.0, 8.0, 16.0])  # constant 100% return each step
        assert ind.rolling_volatility(ind.returns(s), 2).dropna().abs().max() == pytest.approx(0.0)
