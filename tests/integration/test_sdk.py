"""Integration tests for TradingSDK (B16/B20/B23/C4 — the §4 facade)."""

import numpy as np
import pandas as pd
import pytest

from tradedqn.config import load_config
from tradedqn.data.client import DataClient
from tradedqn.sdk import TradingSDK

CONFIG = "config/config.yaml"


def synthetic_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = np.maximum(100 + np.cumsum(rng.normal(0, 1, n)), 1.0)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + rng.uniform(0, 2, n),
            "Low": np.maximum(close - rng.uniform(0, 2, n), 0.5),
            "Close": close,
            "Volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=idx,
    )


def build_sdk(cache_dir):
    client = DataClient(cache_dir=str(cache_dir), fetch_fn=lambda *a: synthetic_ohlcv())
    return TradingSDK(cfg=load_config(CONFIG), data_client=client)


@pytest.fixture
def prepared(tmp_path):
    sdk = build_sdk(tmp_path / "cache")
    sdk.prepare_data()
    return sdk


class TestPipeline:
    def test_prepare_returns_nonempty_splits(self, prepared):
        sizes = prepared._splits
        assert all(len(frame) > prepared.cfg.features.window_size for frame, _ in sizes.values())

    def test_features_normalized_prices_raw(self, prepared):
        feats, prices = prepared._splits["train"]
        assert feats.to_numpy().min() >= 0.0 and feats.to_numpy().max() <= 1.0
        assert prices.max() > 1.5  # raw dollar prices, not normalized

    def test_train_then_backtest_then_recommend(self, prepared):
        history = prepared.train(episodes=1)
        assert len(history) == 1
        result = prepared.backtest()
        assert "equity_curve" in result and np.isfinite(result["total_return"])
        rec = prepared.recommend()
        assert rec["action"] in ("sell", "hold", "buy") and len(rec["q_values"]) == 3

    def test_train_forwards_progress_callback(self, prepared):
        seen: list[int] = []
        prepared.train(episodes=2, on_episode=lambda record: seen.append(record["episode"]))
        assert seen == [0, 1]  # SDK relays each episode to the GUI status bar

    def test_train_rejects_zero_episodes(self, prepared):
        with pytest.raises(ValueError, match="episodes must be"):  # else callers hit a bare IndexError
            prepared.train(episodes=0)

    def test_recommend_on_short_split_raises_clear_error(self, tmp_path):
        client = DataClient(cache_dir=str(tmp_path / "c"), fetch_fn=lambda *a: synthetic_ohlcv(n=45))
        sdk = TradingSDK(cfg=load_config(CONFIG), data_client=client)
        sdk.prepare_data()  # ~26 feature rows → test split « window(30)
        with pytest.raises(ValueError, match="to recommend"):  # clear error, not an opaque torch shape crash
            sdk.recommend("test")

    def test_compare_trains_both_architectures(self, prepared):
        histories = prepared.compare(episodes=1)
        assert set(histories) == {"Dueling DQN", "Plain DQN"}
        assert len(histories["Plain DQN"]) == 1 and "reward" in histories["Plain DQN"][0]

    def test_cfg_with_dueling_toggles_the_flag(self, prepared):
        assert prepared._cfg_with_dueling(True).network.dueling is True   # §9 ablation plumbing
        assert prepared._cfg_with_dueling(False).network.dueling is False  # the two arms genuinely differ

    def test_prepare_data_accepts_ticker_and_date_overrides(self, tmp_path):
        seen: dict[str, str] = {}

        def fetch(ticker, start, end, interval):
            seen.update(ticker=ticker, start=start, end=end)
            return synthetic_ohlcv()

        client = DataClient(cache_dir=str(tmp_path / "ovr"), fetch_fn=fetch)
        sdk = TradingSDK(cfg=load_config(CONFIG), data_client=client)
        sdk.prepare_data(ticker="MSFT", start="2019-01-01", end="2020-01-01")
        assert seen == {"ticker": "MSFT", "start": "2019-01-01", "end": "2020-01-01"}


class TestGuards:
    def test_default_construction_builds_data_client(self):
        sdk = TradingSDK(cfg=load_config(CONFIG))  # no injected client → builds default + gatekeeper
        assert isinstance(sdk.data_client, DataClient)

    def test_requires_prepare_first(self, tmp_path):
        with pytest.raises(RuntimeError, match="prepare_data"):
            build_sdk(tmp_path / "c").train(episodes=1)

    def test_save_load_reproduces_recommendation(self, prepared, tmp_path):
        prepared.train(episodes=1)
        path = str(tmp_path / "brain.pt")
        prepared.save_brain(path)
        rec_before = prepared.recommend()
        fresh = build_sdk(tmp_path / "cache2")
        fresh.prepare_data()
        fresh.load_brain(path)
        assert fresh.recommend()["action_index"] == rec_before["action_index"]

    def test_path_guard_refuses_escape(self, prepared):
        with pytest.raises(ValueError, match="outside the project root"):
            prepared.save_brain("../../../etc/evil.pt")
