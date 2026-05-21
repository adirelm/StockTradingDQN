"""DataClient — fetch & cache OHLCV market data (cache-first, gatekept).

Returns the local CSV cache when present (offline + reproducible). Only on a
cache miss does it call the live fetcher, and that call goes through the §5
gatekeeper. The fetcher is injectable so unit tests never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from tradedqn.data.gatekeeper import RateLimitGatekeeper

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _yf_download(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf

    frame = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if hasattr(frame.columns, "levels"):  # single-ticker MultiIndex (price, ticker) → flatten
        frame.columns = frame.columns.get_level_values(0)
    return frame


class DataClient:
    """Cache-first OHLCV provider with an injected rate-limit gatekeeper."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        gatekeeper: RateLimitGatekeeper | None = None,
        fetch_fn: Callable[[str, str, str, str], pd.DataFrame] = _yf_download,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.gatekeeper = gatekeeper or RateLimitGatekeeper()
        self._fetch_fn = fetch_fn

    def _cache_path(self, ticker: str, start: str, end: str, interval: str) -> Path:
        return self.cache_dir / f"{ticker}_{start}_{end}_{interval}.csv"

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return OHLCV for ``ticker``, cache-first; fetch (gatekept) on a miss."""
        path = self._cache_path(ticker, start, end, interval)
        if path.exists() and not force_refresh:
            return self._read_cache(path)
        # All external calls go through the §5 gatekeeper (throttle + retry + log).
        raw = self.gatekeeper.execute(self._fetch_fn, ticker, start, end, interval)
        frame = self._validate(raw)
        self._write_cache(path, frame)
        return frame

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            raise ValueError("fetcher returned no data")
        missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"missing OHLCV columns: {missing}")
        return frame[OHLCV_COLUMNS].dropna()

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    @staticmethod
    def _write_cache(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
