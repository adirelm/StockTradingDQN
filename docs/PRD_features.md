# PRD — Phase 2: Feature Engineering (indicators + normalization + split)

**Phase:** 2 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B3, B4, B5** (and feeds B6 windowing in Phase 3).

## Goal
Turn raw OHLCV into the normalized **market** features the DQN state is built
from, split chronologically with **no look-ahead / no test-set leakage**.

## Key design decision — 8 market features here, 2 portfolio features later
The deck's 10-feature state = **8 market features** (computable from OHLCV) +
**2 portfolio features** (`position`, `cash_exposure`). The portfolio features
are path-dependent on the agent's own actions, so they are **injected by the
`TradingEnvironment` at runtime (Phase 3)** — precomputing them would be both
meaningless and a leak. Phase 2 therefore produces the **8 market columns**
(config `features.names[:8]`):

| # | feature | formula (sketch) |
|---|---|---|
| 1 | price_return | `Close.pct_change()` |
| 2 | normalized_price | rolling min-max position of Close over `ma_period` |
| 3 | high_low_range | `(High − Low) / Close` |
| 4 | volume_change | `Volume.pct_change()` |
| 5 | ratio_to_ma | `Close / SMA(Close, ma_period) − 1` |
| 6 | volatility | `price_return.rolling(volatility_window).std()` |
| 7 | rsi | Wilder RSI over `rsi_period`, scaled to [0,1] |
| 8 | macd | `EMA(macd_fast) − EMA(macd_slow)` of Close |

## Scope (this phase)
1. **`features/indicators.py`** — pure functions for the math (returns, SMA,
   EMA, RSI, MACD, volatility, ranges). No I/O, fully unit-testable on toy series.
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
indicators: rsi(series, period), macd(series, fast, slow),
            sma(series, n), ema(series, n), rolling_volatility(returns, n), ...
Preprocessor(cfg).compute(ohlcv: DataFrame) -> DataFrame   # 8 market columns
chronological_split(df, train, val, test) -> (train, val, test)
MinMaxNormalizer().fit(train_df) -> self ; .transform(df) -> DataFrame
```

## Acceptance criteria (tests assert)
- Indicators: RSI ∈ [0,100] (then scaled), all-rising series → RSI→100; MACD
  sign tracks fast-vs-slow; SMA/EMA match hand-computed values on a toy series;
  output length equals input (NaNs at warmup, not dropped inside the math fns).
- Preprocessor: output has exactly the 8 market columns in config order; no NaNs
  after warmup drop; row count = input − warmup.
- Split: 70/15/15 by count, **contiguous & time-ordered** (train ends before
  val starts before test); ratios validated to sum ≈ 1.
- Normalizer: after `fit(train).transform(train)` every value ∈ [0,1]; val/test
  transformed with **train** min/max (a test value above train-max clips to 1.0)
  — proves no leakage.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
