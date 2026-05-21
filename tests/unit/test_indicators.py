"""Tests for the technical-indicator math (§4 — pure functions on toy series)."""

import numpy as np
import pandas as pd
import pytest

from tradedqn.features import indicators as ind


@pytest.fixture
def rising():
    return pd.Series([float(i) for i in range(1, 21)])  # 1..20, strictly increasing


@pytest.fixture
def falling():
    return pd.Series([float(i) for i in range(20, 0, -1)])  # 20..1, strictly decreasing


class TestReturnsAndAverages:
    def test_log_return_matches_log_ratio(self):
        s = pd.Series([100.0, 110.0])
        assert ind.log_return(s).iloc[1] == pytest.approx(np.log(110.0 / 100.0))

    def test_sma_matches_hand_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert ind.sma(s, 2).tolist()[1:] == [1.5, 2.5, 3.5]

    def test_ema_first_value_equals_seed(self):
        s = pd.Series([10.0, 20.0, 30.0])
        assert ind.ema(s, 2).iloc[0] == 10.0  # adjust=False seeds on the first point


class TestRSI:
    def test_all_rising_rsi_is_100(self, rising):
        assert ind.rsi(rising, 5).dropna().iloc[-1] == pytest.approx(100.0)

    def test_all_falling_rsi_is_0(self, falling):
        assert ind.rsi(falling, 5).dropna().iloc[-1] == pytest.approx(0.0)

    def test_rsi_bounded(self):
        s = pd.Series([1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13], dtype=float)
        assert ind.rsi(s, 4).dropna().between(0.0, 100.0).all()


class TestMacd:
    def test_macd_positive_when_trending_up(self, rising):
        assert ind.macd(rising, 3, 6).dropna().iloc[-1] > 0

    def test_macd_negative_when_trending_down(self, falling):
        assert ind.macd(falling, 3, 6).dropna().iloc[-1] < 0

    def test_macd_signal_is_ema_of_line(self, rising):
        line = ind.macd(rising, 3, 6)
        assert ind.macd_signal(line, 4).equals(ind.ema(line, 4))


class TestBandsVolumeVwap:
    def test_bollinger_pct_half_at_mean(self):
        s = pd.Series([10.0, 12.0, 8.0, 11.0, 9.0, 10.0])  # last point sits at the SMA
        bb = ind.bollinger_pct(s, 5)
        assert bb.dropna().iloc[-1] == pytest.approx(0.5, abs=0.05)

    def test_vwap_dist_zero_on_flat_price(self):
        flat = pd.Series([10.0] * 6)
        vol = pd.Series([100.0] * 6)
        assert ind.vwap_dist(flat, flat, flat, vol, 3).dropna().abs().max() == pytest.approx(0.0)

    def test_volume_norm_zero_for_constant_volume(self):
        vol = pd.Series([500.0] * 8)
        assert ind.volume_norm(vol, 3).dropna().abs().max() == pytest.approx(0.0)
