# PRD — Phase 7: TradingSDK (the single facade)

**Phase:** 7 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B16, B20, B23** and **C4** (§4 SDK mandate). The architectural keystone.

## Goal
One object that wires the whole pipeline behind a clean API. The UIs
(terminal, GUI) call **only** the SDK — never the engine modules directly.

## Pipeline (deck: Prepare Data → Train → Backtest → Predict)
```
prepare_data()  DataClient → Preprocessor (8 market feats) → chronological_split
                → MinMaxNormalizer.fit(TRAIN).transform(all) → store
                  {train,validation,test}: (normalized features, raw close prices)
train(episodes) build TradingEnvironment on TRAIN → TrainingService → history
backtest(split) build env on the split (default TEST/held-out) → BacktestService
recommend(split) latest 30×10 window (flat portfolio) → InferenceService
save_brain / load_brain  agent checkpoint, path-guarded (§13) + weights_only (§7)
```

## Design notes
- **Prices stay raw** (real dollars for portfolio value); only **features** are
  normalized. Fit the normalizer on TRAIN only (no leakage — carries Phase 2).
- DRY: the 30×10 state assembly (market window + 2 portfolio channels) is a
  shared `env.assemble_state(...)` helper used by both the env and the SDK's
  inference path.
- `recommend` builds the latest state assuming a **flat** portfolio
  (position 0, unrealized_pnl 0) → "what to do if currently in cash".
- Injectable `data_client` / `agent` so tests never hit the network.

## Public API
```
TradingSDK(config_path=None, cfg=None, data_client=None, agent=None)
  .prepare_data() -> {split: n_rows}
  .train(episodes=None) -> list[dict]      # defaults to cfg.training.episodes
  .backtest(split="test") -> dict
  .recommend(split="test") -> {action, action_index, q_values}
  .save_brain(path); .load_brain(path)
```

## Acceptance criteria (tests assert; injected fake fetcher, no network)
- `prepare_data` returns non-empty train/validation/test sizes; calling train/
  backtest before it raises a clear error.
- splits are chronological & disjoint; features normalized in [0,1], prices raw.
- `train(episodes=1)` returns a 1-element history; `backtest()` returns the full
  summary dict; `recommend()` returns a valid action + 3 Q-values.
- `save_brain`→`load_brain` into a fresh SDK reproduces the greedy recommendation.
- path guard: a relative path escaping the project root is refused.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
