"""Tests for the Preprocessor (B3 — 8 market features in config order, no NaNs)."""

import numpy as np
import pandas as pd
import pytest

from tradedqn.config import load_config
from tradedqn.features.preprocessor import MARKET_FEATURES, Preprocessor


@pytest.fixture
def features_cfg():
    return load_config("config/config.yaml").features


@pytest.fixture
def ohlcv():
    n = 120
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": (close + rng.uniform(0, 2, n)).values,
            "Low": (close - rng.uniform(0, 2, n)).values,
            "Close": close.values,
            "Volume": rng.integers(1_000, 5_000, n).astype(float),
        },
        index=idx,
    )


class TestPreprocessor:
    def test_exactly_eight_market_columns_in_order(self, features_cfg, ohlcv):
        out = Preprocessor(features_cfg).compute(ohlcv)
        assert list(out.columns) == MARKET_FEATURES
        assert len(MARKET_FEATURES) == 8

    def test_no_nans_after_warmup_drop(self, features_cfg, ohlcv):
        out = Preprocessor(features_cfg).compute(ohlcv)
        assert not out.isna().any().any()
        assert len(out) < len(ohlcv)  # warmup rows were dropped

    def test_rsi_column_scaled_to_unit_range(self, features_cfg, ohlcv):
        out = Preprocessor(features_cfg).compute(ohlcv)
        assert out["rsi"].between(0.0, 1.0).all()

    def test_market_features_are_first_eight_config_names(self, features_cfg):
        assert list(features_cfg.names[:8]) == MARKET_FEATURES
