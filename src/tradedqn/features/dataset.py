"""Chronological split + fit-on-train normalizer.

Both exist to prevent **look-ahead / test-set leakage** — the cardinal sin of
financial ML. The split is time-ordered (never shuffled); the normalizer learns
its min/max from the **train** slice only and clips val/test into that range.
"""

from __future__ import annotations

import pandas as pd


def chronological_split(
    frame: pd.DataFrame, train: float, validation: float, test: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by time into contiguous train / validation / test slices.

    Rows keep their order (no shuffle), so the test slice is strictly *after*
    train and validation. Ratios must sum to ~1.0.
    """
    total = train + validation + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    n = len(frame)
    n_train = int(n * train)
    n_val = int(n * validation)
    return (
        frame.iloc[:n_train],
        frame.iloc[n_train : n_train + n_val],
        frame.iloc[n_train + n_val :],
    )


class MinMaxNormalizer:
    """Min-max scaler to [0, 1], fit on the training slice only."""

    def __init__(self) -> None:
        self._min: pd.Series | None = None
        self._range: pd.Series | None = None

    def fit(self, train_frame: pd.DataFrame) -> MinMaxNormalizer:
        """Learn per-column min and (max − min) from the training data."""
        self._min = train_frame.min()
        span = train_frame.max() - self._min
        self._range = span.replace(0.0, 1.0)  # constant column → leave as-is (→ 0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale with the train stats, clipped to [0, 1] (no leakage from val/test)."""
        if self._min is None or self._range is None:
            raise RuntimeError("MinMaxNormalizer.transform called before fit")
        scaled = (frame - self._min) / self._range
        return scaled.clip(lower=0.0, upper=1.0)

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_frame).transform(train_frame)
