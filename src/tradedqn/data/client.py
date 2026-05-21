"""DataClient — fetch & cache OHLCV market data (cache-first, gatekept).

Returns the local parquet cache when present (offline + reproducible). Only on a
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
    """Fetch OHLCV from Yahoo Finance, flattening single-ticker MultiIndex columns."""
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

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        """Parquet cache path: ``{cache_dir}/{ticker}_{start}_{end}.parquet`` (§4)."""
        return self.cache_dir / f"{ticker}_{start}_{end}.parquet"

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """OHLCV for ``ticker``: parquet cache first, else a gatekept live fetch,
        else the ``{ticker}.csv`` fallback if the fetch fails (§4)."""
        path = self._cache_path(ticker, start, end)
        if path.exists() and not force_refresh:
            return self._read_cache(path)
        try:  # all external calls go through the §5 gatekeeper (throttle + retry + log)
            raw = self.gatekeeper.execute(self._fetch_fn, ticker, start, end, interval)
            frame = self._validate(raw)
        except Exception as error:  # network/HTTP/empty → CSV fallback if present
            return self._read_fallback(ticker, error)
        self._write_cache(path, frame)
        return frame

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        """Require non-empty OHLCV columns; return the cleaned frame (NaNs dropped)."""
        if frame is None or frame.empty:
            raise ValueError("fetcher returned no data")
        missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"missing OHLCV columns: {missing}")
        return frame[OHLCV_COLUMNS].dropna()

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        """Read the cached OHLCV parquet."""
        return pd.read_parquet(path)

    def _read_fallback(self, ticker: str, error: Exception) -> pd.DataFrame:
        """CSV fallback ``{cache_dir}/{ticker}.csv`` (Date index) when a live fetch fails."""
        fallback = self.cache_dir / f"{ticker}.csv"
        if not fallback.exists():
            raise error
        return self._validate(pd.read_csv(fallback, index_col="Date", parse_dates=True))

    @staticmethod
    def _write_cache(path: Path, frame: pd.DataFrame) -> None:
        """Write an OHLCV frame to the parquet cache (snappy; creating the directory)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, compression="snappy")
