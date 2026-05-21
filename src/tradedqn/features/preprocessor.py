"""Preprocessor — assemble the brief's 8 market features from OHLCV (§4).

Features (config order): ``log_return, rsi_14, macd, macd_signal, macd_hist,
bb_pct, vwap_dist, volume_norm``. The two remaining state channels
(``position``, ``unrealized_pnl``) are portfolio-dependent and injected by the
``TradingEnvironment`` at runtime — precomputing them would be meaningless and a leak.
"""

from __future__ import annotations

import pandas as pd

from tradedqn.features import indicators as ind

# The 8 market features, in config `features.names[:8]` order.
MARKET_FEATURES = [
    "log_return",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "vwap_dist",
    "volume_norm",
]


class Preprocessor:
    """Builds the market-feature frame from OHLCV using ``config.features`` params."""

    def __init__(self, features_cfg) -> None:
        self.cfg = features_cfg

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """OHLCV → DataFrame with the 8 market columns; warmup NaN rows dropped."""
        c = self.cfg
        close, high, low, volume = (ohlcv[x] for x in ("Close", "High", "Low", "Volume"))
        macd_line = ind.macd(close, c.macd_fast, c.macd_slow)
        signal = ind.macd_signal(macd_line, c.macd_signal)
        columns = {
            "log_return": ind.log_return(close),
            "rsi_14": ind.rsi(close, c.rsi_period),
            "macd": macd_line,
            "macd_signal": signal,
            "macd_hist": macd_line - signal,
            "bb_pct": ind.bollinger_pct(close, c.ma_period, c.bb_num_std),
            "vwap_dist": ind.vwap_dist(high, low, close, volume, c.ma_period),
            "volume_norm": ind.volume_norm(volume, c.ma_period),
        }
        frame = pd.DataFrame(columns, index=ohlcv.index)[MARKET_FEATURES]
        return frame.dropna()
