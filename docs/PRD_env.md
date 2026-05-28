# PRD — Phase 3: TradingEnvironment (state, actions, reward)

**Phase:** 3 of 10 · **Status:** approved-to-build · built & shipped (v1.0.0) · Covers requirement rows
**B6, B7, B10, B18** (state) and feeds the agent (Phase 4).

## Goal
A Gym-style environment that turns the normalized market features + portfolio
state into the **30×10** observation, applies **Sell/Hold/Buy**, and returns the
**risk/cost-adjusted reward** `rₜ = ΔVₜ − Cₜ − Sₜ + λ·Sharpeₜ`.

## Modules
- `env/portfolio.py` — `Portfolio`: cash + shares accounting; `buy/sell/hold`
  at a price with transaction-cost + slippage fees; `value(price)`,
  `position(price)`, `unrealized_pnl(price)`.
- `env/reward.py` — `RewardFunction`: composes the deck's reward and exposes the
  components (`delta_v`, `cost`, `slippage`, `sharpe`) in `info` for transparency.
- `env/trading_env.py` — `TradingEnvironment`: `reset() -> state`,
  `step(action) -> (state, reward, done, info)`, 30×10 assembly.

## Design decisions (need human sign-off — §1.4 reward/architecture)

**D1 — Position sizing: all-in / all-out (binary).** Buy = invest *all* cash;
Sell = liquidate *all* holdings; Hold = no change. This is the deck's simple
"have stock or not" model. *Alternative:* fractional sizing (trade in fixed %
increments) — more realistic, more state, harder to learn at this budget.

**D2 — Portfolio features in the window.** `position` and `unrealized_pnl` are
the agent's *current* scalars, **broadcast across all 30 rows** of the window
(constant channels 9 & 10; channels 1–8 are the per-day market history).
*Alternative:* store the per-day portfolio *path* over the window
(path-dependent, more realistic, more bookkeeping).

**D3 — Reward composition.** Keep the deck's full formula
`rₜ = ΔVₜ − Cₜ − Sₜ + λ·Sharpeₜ`, units = fraction of initial capital:
- `ΔVₜ` = mark-to-market portfolio value change over the step,
- `Cₜ` = `transaction_cost × traded notional`, `Sₜ` = `slippage × traded notional`,
- `Sharpeₜ` = rolling mean/std of recent step returns over `sharpe_window`, `λ=risk_lambda`.
*Alternative for a first cut:* drop the Sharpe term (`λ=0`) → pure net-return
reward; add risk shaping later in experiments.

## No-look-ahead contract
No future look-ahead: at decision day `t` the state window ends at **and
includes** day `t` (feature rows `t−window+1 … t`), and the trade executes at
`prices[t]` — the bar the agent observes. The *next* day's price (`prices[t+1]`)
is the step's outcome, never part of the observed state. A test asserts the
window's last row is day `t`'s row, never the not-yet-observed `t+1` row.

## Public API
```
Portfolio(initial_capital, transaction_cost, slippage)
  .buy(price)/.sell(price)/.hold(); .value(price); .position(price); .unrealized_pnl(price)
RewardFunction(risk_lambda, sharpe_window).compute(...) -> (reward, components)
TradingEnvironment(features_df, prices, cfg)
  .reset() -> ndarray(30,10);  .step(action) -> (ndarray(30,10), float, bool, dict)
```

## Acceptance criteria (tests assert)
- Portfolio: buy spends all cash minus fees; sell liquidates; fees match config;
  value = cash + shares·price; unrealized_pnl is 0 when flat, signed when holding.
- Env: `reset` returns float32 (30,10); channels 9–10 reflect portfolio; `step`
  advances one day, `done` true at the end; **no-look-ahead** test passes;
  buying then price-up yields positive reward; a Hold incurs no cost.
- Reward components sum to the returned reward; `λ=0` removes the Sharpe term.

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean.
