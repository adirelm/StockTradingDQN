# PRD — Phase 7: TradingSDK (the single facade)

**Phase:** 7 of 10 · **Status:** approved-to-build · Covers requirement rows
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
compare(episodes) train Dueling + plain DQN on TRAIN → {arch: history} (§9 ablation)
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
  .prepare_data(ticker=None, start=None, end=None) -> {split: n_rows}
                                           # None keeps the config default; pass to override symbol/range
  .train(episodes=None, on_episode=None) -> list[dict]
                                           # episodes defaults to cfg.training.episodes;
                                           # on_episode(record) streams each per-episode dict (GUI live training)
  .backtest(split="test") -> dict
  .compare(episodes=None, on_episode=None) -> {arch_name: list[dict]}
                                           # §9 Dueling-vs-plain ablation: trains both arches on the same data;
                                           # on_episode(arch_name, record) streams live progress (GUI/CLI button)
  .recommend(split="test") -> {action, action_index, q_values,
                               names, confidence, top_features}
                                           # confidence = softmax over Q; top_features = saliency-ranked channels (§8)
  .save_brain(path, metadata=None); .load_brain(path)
```

## Acceptance criteria (tests assert; injected fake fetcher, no network)
- `prepare_data` returns non-empty train/validation/test sizes; calling train/
  backtest before it raises a clear error.
- splits are chronological & disjoint; features normalized in [0,1], prices raw.
- `train(episodes=1)` returns a 1-element history; `backtest()` returns the full
  summary dict; `recommend()` returns a valid action + 3 Q-values plus its
  confidence (softmax) and the top saliency-ranked features (§8).
- `save_brain`→`load_brain` into a fresh SDK reproduces the greedy recommendation.
- path guard: a relative path escaping the project root is refused.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
