"""Technical-indicator math — pure functions over pandas Series.

No I/O and no config: each takes a Series (and parameters) and returns a Series
of the same length (NaN during warmup), so they unit-test on toy data. Together
they build the brief's 8 market features: ``log_return, rsi_14, macd,
macd_signal, macd_hist, bb_pct, vwap_dist, volume_norm`` (§4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series) -> pd.Series:
    """Log return ln(Pₜ / Pₜ₋₁) — additive over time, symmetric around 0."""
    return np.log(close).diff()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (no warmup bias: ``adjust=False``)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index in [0, 100] (simple-average variant).

    A monotonically rising series has zero losses → RSI → 100; a falling one → 0.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return out.astype(float)


def macd(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """MACD line = EMA(fast) − EMA(slow). Positive ⇒ short-term momentum up."""
    return ema(close, fast) - ema(close, slow)


def macd_signal(macd_line: pd.Series, signal: int) -> pd.Series:
    """Signal line = EMA(signal) of the MACD line."""
    return ema(macd_line, signal)


def bollinger_pct(close: pd.Series, window: int, num_std: float = 2.0) -> pd.Series:
    """Bollinger %B — close's position within its ±k·σ band (≈0 at lower … 1 at upper)."""
    mid = sma(close, window)
    sd = close.rolling(window).std()
    lower = mid - num_std * sd
    span = (2.0 * num_std * sd).replace(0.0, np.nan)
    return ((close - lower) / span).astype(float)


def vwap_dist(high, low, close, volume, window: int) -> pd.Series:
    """Relative distance of close from the rolling VWAP: ``Close / VWAP − 1``."""
    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(window).sum()
    vol = volume.rolling(window).sum().replace(0.0, np.nan)
    return (close / (pv / vol) - 1.0).astype(float)


def volume_norm(volume: pd.Series, window: int) -> pd.Series:
    """Volume relative to its moving average: ``Volume / SMA(Volume) − 1``."""
    avg = sma(volume, window).replace(0.0, np.nan)
    return (volume / avg - 1.0).astype(float)
