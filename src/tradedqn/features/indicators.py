"""Technical-indicator math — pure functions over pandas Series.

No I/O and no config: each takes a Series (and parameters) and returns a Series
of the same length (NaN during warmup). Kept pure so they unit-test on toy data.
"""

from __future__ import annotations

import pandas as pd


def returns(series: pd.Series) -> pd.Series:
    """Simple period-over-period return."""
    return series.pct_change()


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
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 (only gains) → RSI 100; avg_gain == 0 (only losses) → RSI 0
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    return out.astype(float)


def macd(series: pd.Series, fast: int, slow: int) -> pd.Series:
    """MACD line = EMA(fast) − EMA(slow). Positive ⇒ short-term momentum up."""
    return ema(series, fast) - ema(series, slow)


def rolling_volatility(return_series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of returns — local risk/uncertainty."""
    return return_series.rolling(window).std()


def high_low_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Intraday range normalised by close: ``(High − Low) / Close``."""
    return (high - low) / close


def ratio_to_ma(close: pd.Series, window: int) -> pd.Series:
    """Relative deviation of price from its moving average: ``Close / SMA − 1``."""
    return close / sma(close, window) - 1.0


def normalized_price(close: pd.Series, window: int) -> pd.Series:
    """Relative position of close within its rolling [min, max] window → ~[0, 1]."""
    lo = close.rolling(window).min()
    hi = close.rolling(window).max()
    span = (hi - lo).replace(0.0, pd.NA)
    return ((close - lo) / span).astype(float)
