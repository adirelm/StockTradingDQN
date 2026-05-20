"""Tests for chronological split + fit-on-train normalizer (B5 — no leakage)."""

import pandas as pd
import pytest

from tradedqn.features.dataset import MinMaxNormalizer, chronological_split


@pytest.fixture
def frame():
    return pd.DataFrame({"a": range(100), "b": range(100, 200)}, dtype=float)


class TestChronologicalSplit:
    def test_counts_70_15_15(self, frame):
        tr, va, te = chronological_split(frame, 0.70, 0.15, 0.15)
        assert (len(tr), len(va), len(te)) == (70, 15, 15)

    def test_slices_are_contiguous_and_time_ordered(self, frame):
        tr, va, te = chronological_split(frame, 0.70, 0.15, 0.15)
        assert tr.index.max() < va.index.min() < te.index.min()
        assert te.index.max() == frame.index.max()

    def test_ratios_must_sum_to_one(self, frame):
        with pytest.raises(ValueError, match="sum to 1.0"):
            chronological_split(frame, 0.6, 0.2, 0.1)


class TestMinMaxNormalizer:
    def test_train_transform_in_unit_range(self, frame):
        tr, _, _ = chronological_split(frame, 0.70, 0.15, 0.15)
        scaled = MinMaxNormalizer().fit_transform(tr)
        assert scaled.min().min() == pytest.approx(0.0)
        assert scaled.max().max() == pytest.approx(1.0)

    def test_uses_train_stats_and_clips_future_highs(self, frame):
        tr, _, te = chronological_split(frame, 0.70, 0.15, 0.15)
        norm = MinMaxNormalizer().fit(tr)
        out = norm.transform(te)
        # test values exceed train max → must clip to 1.0 (proves no leakage)
        assert (out == 1.0).all().all()

    def test_transform_before_fit_raises(self, frame):
        with pytest.raises(RuntimeError, match="before fit"):
            MinMaxNormalizer().transform(frame)

    def test_constant_column_maps_to_zero(self):
        df = pd.DataFrame({"flat": [5.0, 5.0, 5.0]})
        out = MinMaxNormalizer().fit_transform(df)
        assert (out["flat"] == 0.0).all()
