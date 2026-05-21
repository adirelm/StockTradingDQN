"""Tests for DataClient (B1/B2 — cache-first, gatekept, injected fetcher)."""

import pandas as pd
import pytest

from tradedqn.data.client import OHLCV_COLUMNS, DataClient


class SpyGatekeeper:
    def __init__(self):
        self.executed = 0

    def execute(self, api_call, *args, **kwargs):
        self.executed += 1
        return api_call(*args, **kwargs)


def make_frame(rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({c: range(rows) for c in OHLCV_COLUMNS}, index=idx)


class CountingFetcher:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = 0

    def __call__(self, ticker, start, end, interval):
        self.calls += 1
        return self.frame


@pytest.fixture
def client_parts(tmp_path):
    fetcher = CountingFetcher(make_frame())
    gk = SpyGatekeeper()
    client = DataClient(cache_dir=str(tmp_path), gatekeeper=gk, fetch_fn=fetcher)
    return client, fetcher, gk


class TestCacheFirst:
    def test_cache_miss_fetches_through_gatekeeper_and_writes(self, client_parts, tmp_path):
        client, fetcher, gk = client_parts
        out = client.get_ohlcv("AAPL", "2020-01-01", "2020-01-06")
        assert fetcher.calls == 1
        assert gk.executed == 1
        assert list(out.columns) == OHLCV_COLUMNS
        assert (tmp_path / "AAPL_2020-01-01_2020-01-06_1d.csv").exists()

    def test_cache_hit_does_not_fetch(self, client_parts):
        client, fetcher, gk = client_parts
        client.get_ohlcv("AAPL", "2020-01-01", "2020-01-06")
        client.get_ohlcv("AAPL", "2020-01-01", "2020-01-06")
        assert fetcher.calls == 1  # second call served from cache

    def test_force_refresh_refetches(self, client_parts):
        client, fetcher, _ = client_parts
        client.get_ohlcv("AAPL", "2020-01-01", "2020-01-06")
        client.get_ohlcv("AAPL", "2020-01-01", "2020-01-06", force_refresh=True)
        assert fetcher.calls == 2


class TestValidation:
    def test_empty_frame_raises(self, tmp_path):
        client = DataClient(cache_dir=str(tmp_path), gatekeeper=SpyGatekeeper(),
                            fetch_fn=lambda *a: pd.DataFrame())
        with pytest.raises(ValueError, match="no data"):
            client.get_ohlcv("X", "2020-01-01", "2020-01-02")

    def test_missing_column_raises(self, tmp_path):
        bad = make_frame().drop(columns=["Volume"])
        client = DataClient(cache_dir=str(tmp_path), gatekeeper=SpyGatekeeper(),
                            fetch_fn=lambda *a: bad)
        with pytest.raises(ValueError, match="missing OHLCV columns"):
            client.get_ohlcv("X", "2020-01-01", "2020-01-02")
