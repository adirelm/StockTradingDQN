# PRD — Phase 2: Feature Engineering (indicators + normalization + split)

**Phase:** 2 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B3, B4, B5** (and feeds B6 windowing in Phase 3).

## Goal
Turn raw OHLCV into the normalized **market** features the DQN state is built
from, split chronologically with **no look-ahead / no test-set leakage**.

## Key design decision — 8 market features here, 2 portfolio features later
The brief's 10-feature state (§4) = **8 market features** (computable from OHLCV) +
**2 portfolio features** (`position`, `unrealized_pnl`). The portfolio features
are path-dependent on the agent's own actions, so they are **injected by the
`TradingEnvironment` at runtime (Phase 3)** — precomputing them would be both
meaningless and a leak. Phase 2 therefore produces the **8 market columns**
(config `features.names[:8]`):

| # | feature | formula (sketch) |
|---|---|---|
| 1 | log_return | `ln(Close).diff()` |
| 2 | rsi_14 | Wilder RSI over `rsi_period` (raw 0–100; the normalizer scales it) |
| 3 | macd | `EMA(macd_fast) − EMA(macd_slow)` of Close |
| 4 | macd_signal | `EMA(macd_signal)` of the MACD line |
| 5 | macd_hist | `macd − macd_signal` |
| 6 | bb_pct | Bollinger %B over `ma_period` (±`bb_num_std`·σ band) |
| 7 | vwap_dist | `Close / rolling-VWAP(ma_period) − 1` |
| 8 | volume_norm | `Volume / SMA(Volume, ma_period) − 1` |

## Scope (this phase)
1. **`features/indicators.py`** — pure functions for the math (log_return, SMA,
   EMA, RSI, MACD + signal, Bollinger %B, VWAP distance, volume-norm). No I/O,
   fully unit-testable on toy series.
2. **`features/preprocessor.py`** — `Preprocessor.compute(ohlcv) -> DataFrame`:
   assemble the 8 columns (in config order), drop warmup NaN rows.
3. **`features/dataset.py`** —
   - `chronological_split(df, train, val, test)` → 3 frames, time-ordered, no shuffle.
   - `MinMaxNormalizer.fit(train_df)` then `.transform(df)` → values in [0,1],
     **fit on train only**, val/test transformed with train stats and clipped.

## Out of scope (later)
- 30×10 window assembly + the 2 portfolio features → Phase 3 (env).
- The agent / network → Phase 4.

## Public API
```
indicators: log_return(close), rsi(series, period), macd(close, fast, slow),
            macd_signal(line, n), bollinger_pct(close, n, k), vwap_dist(h,l,c,v,n), volume_norm(v,n)
Preprocessor(cfg).compute(ohlcv: DataFrame) -> DataFrame   # 8 market columns
chronological_split(df, train, val, test) -> (train, val, test)
MinMaxNormalizer().fit(train_df) -> self ; .transform(df) -> DataFrame
```

## Acceptance criteria (tests assert)
- Indicators: RSI ∈ [0,100], all-rising series → RSI→100; MACD sign tracks
  fast-vs-slow; bb_pct ≈ 0.5 at the band mean; vwap_dist / volume_norm ≈ 0 on
  flat input; SMA/EMA match hand-computed values on a toy series.
- Preprocessor: output has exactly the 8 market columns in config order; no NaNs
  after warmup drop; row count = input − warmup.
- Split: 70/15/15 by count, **contiguous & time-ordered** (train ends before
  val starts before test); ratios validated to sum ≈ 1.
- Normalizer: after `fit(train).transform(train)` every value ∈ [0,1]; val/test
  transformed with **train** min/max (a test value above train-max clips to 1.0)
  — proves no leakage.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
