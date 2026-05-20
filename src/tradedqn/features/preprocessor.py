"""Preprocessor — assemble the 8 market features from OHLCV (config order).

The two remaining state features (``position``, ``cash_exposure``) are
portfolio-dependent and are injected by the ``TradingEnvironment`` at runtime
(Phase 3), not computed here — precomputing them would be meaningless and a leak.
"""

from __future__ import annotations

import pandas as pd

from tradedqn.features import indicators as ind

# The 8 market features, in config `features.names[:8]` order.
MARKET_FEATURES = [
    "price_return",
    "normalized_price",
    "high_low_range",
    "volume_change",
    "ratio_to_ma",
    "volatility",
    "rsi",
    "macd",
]


class Preprocessor:
    """Builds the market-feature frame from OHLCV using ``config.features`` params."""

    def __init__(self, features_cfg) -> None:
        self.cfg = features_cfg

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """OHLCV → DataFrame with the 8 market columns; warmup NaN rows dropped."""
        close, high, low, volume = (ohlcv[c] for c in ("Close", "High", "Low", "Volume"))
        daily_return = ind.returns(close)
        columns = {
            "price_return": daily_return,
            "normalized_price": ind.normalized_price(close, self.cfg.ma_period),
            "high_low_range": ind.high_low_range(high, low, close),
            "volume_change": ind.returns(volume),
            "ratio_to_ma": ind.ratio_to_ma(close, self.cfg.ma_period),
            "volatility": ind.rolling_volatility(daily_return, self.cfg.volatility_window),
            "rsi": ind.rsi(close, self.cfg.rsi_period) / 100.0,
            "macd": ind.macd(close, self.cfg.macd_fast, self.cfg.macd_slow),
        }
        frame = pd.DataFrame(columns, index=ohlcv.index)[MARKET_FEATURES]
        return frame.dropna()
