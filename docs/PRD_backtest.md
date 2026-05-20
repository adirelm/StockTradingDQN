# PRD — Phase 6: Backtest + Inference

**Phase:** 6 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B16, B17, B18**. This is the "prove it works" evidence the lecturer stresses.

## Goal
Run the trained **greedy** policy over a held-out slice, report the equity
curve vs a Buy & Hold benchmark with the deck's metrics, and expose single-step
inference (latest window → recommended action).

## Modules
- `services/metrics.py` — pure functions over an equity series:
  `total_return`, `sharpe_ratio` (annualized, std-guarded), `max_drawdown`.
- `services/backtest.py` — `BacktestService(env, agent).run() -> dict`:
  greedy rollout, tracks equity + prices per step, counts trades and computes
  win-rate over completed round-trips, builds the Buy & Hold benchmark.
- `services/inference.py` — `InferenceService(agent, action_names).recommend(state)`
  → `{action, action_index, q_values}` (greedy argmax of the policy Q-vector);
  `action_names(cfg)` helper orders names by their config index.

## Small enabling changes
- `env.step` info gains `price` (next price) and `traded` (notional) so the
  backtest can build the benchmark and count real trades.
- `DQNAgent.q_values(state) -> np.ndarray` exposes the raw Q-vector for inference.

## Metric definitions
- `total_return = equity[-1]/equity[0] − 1`
- `sharpe_ratio = √252 · mean(r)/std(r)` over step returns (`0` if std==0)
- `max_drawdown = max_t (peak_so_far − equity_t)/peak_so_far` (positive fraction)
- `num_trades` = executed Buy/Sell actions (`traded > 0`)
- `win_rate` = winning round-trips / total round-trips (a round-trip = Buy→Sell;
  wins if exit equity > entry equity); `0` if no round-trips. *Documented
  simplification: round-trip P&L, not per-day.*
- Benchmark = `initial_capital · price_t / price_0` (buy-and-hold over the slice).

## Acceptance criteria (tests assert)
- metrics: `total_return([100,110])==0.1`; `max_drawdown([100,120,90,150])==0.25`;
  monotonic-up equity → drawdown 0; constant returns → sharpe 0 (std guard).
- backtest: `run()` returns `equity_curve`, `benchmark_curve` (equal length),
  `total_return`, `sharpe_ratio`, `max_drawdown`, `win_rate`, `num_trades`,
  `benchmark_return`; all finite; `num_trades ≥ 0`; equity starts at initial capital.
- inference: `recommend` returns an action in the names; `q_values` length 3;
  `action_index == argmax(q_values)`; `action_names(cfg) == ['sell','hold','buy']`.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
