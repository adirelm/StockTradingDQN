# PLAN — architecture & design (TradeDQN)

Design/architecture doc (§2.2). Diagrams render in the README; this is the
single place that aggregates architecture, layering, ADRs, interface contracts,
and the concurrency decision.

## Architecture (C4-ish)

**Context.** One user (or AI agent) drives TradeDQN; the only external system is
Yahoo Finance (market data). No DB, no broker.

**Containers / layers** (Consumers → SDK → Services → Domain → Infrastructure):

```
UI:    TerminalApp (cli/menu.py) , MainWindow→GuiController (gui/)
        │  (presentation only; depend ONLY on the SDK)
SDK:    TradingSDK (sdk.py)                      ← single business-logic entry point (§4)
        │
Svc:    TrainingService , BacktestService , InferenceService , metrics
        │
Model:  DQNAgent → DuelingDQN + ReplayBuffer + (target net)
Env:    TradingEnvironment → Portfolio + RewardFunction
Data:   DataClient → RateLimitGatekeeper (§5) + parquet cache ; Preprocessor + split/normalizer
Infra:  Yahoo Finance (yfinance) , local parquet cache , torch checkpoints
```

The README contains the rendered **data-flow** and **OOP-layers** mermaid
diagrams plus the **Dueling Conv1D network** diagram. Dependency rule: arrows
point downward only — a UI never imports an engine module; the SDK is the seam.

## Components (interface contracts)
| Component | Key interface |
|---|---|
| `TradingSDK` | `prepare_data() → {split:n}`, `train(episodes) → history`, `backtest(split) → metrics`, `recommend(split) → {action,q}`, `save_brain/load_brain(path)` |
| `DataClient` | `get_ohlcv(ticker,start,end,interval,force_refresh) → DataFrame` (cache-first, gatekept) |
| `RateLimitGatekeeper` | `acquire(wait=True) → waited_seconds` (min-interval + sliding-window) |
| `TradingEnvironment` | `reset() → state(30×10)`, `step(action) → (state, reward, done, info)` |
| `DQNAgent` | `act(state,greedy) → int`, `remember(...)`, `learn() → loss\|None`, `q_values(state)`, `save/load` |
| `BacktestService` | `run() → {equity_curve, benchmark_curve, total_return, sharpe_ratio, max_drawdown, win_rate, num_trades}` |

## Architecture Decision Records
- **ADR-001** — config format: YAML over the deck's `setup.json` (inline-documented
  hyperparameters; deck parameter *names* kept). See [ADR-001-config-format.md](ADR-001-config-format.md).
- **ADR (PRD_env D1)** — all-in/all-out position model (deck's "have stock or not").
- **ADR (PRD_env D3)** — reward `r = ΔV − C − S + λ·Sharpe` (risk/cost-adjusted, not raw profit).
- **ADR (PRD_gui)** — Tkinter + matplotlib GUI (stdlib + existing dep) over Streamlit.

## Concurrency & thread safety (§15)
The pipeline is **single-threaded by design**. Cost centres: **CPU-bound**
(PyTorch forward/backward in training & inference) and **I/O-bound** (one
rate-limited Yahoo fetch, cache-first → usually zero calls). For a single
sequential RL loop over one symbol at this scale, multiprocessing/threading adds
complexity with no real gain and the GIL is not the bottleneck.
`RateLimitGatekeeper` holds mutable state (a timestamp deque) and is
**single-threaded by contract — not thread-safe**; a future parallel multi-ticker
sweep must give each worker its own gatekeeper or guard `acquire`/`execute` with
a lock, and use a `multiprocessing.Pool` for the embarrassingly-parallel
per-ticker training.

## Build sequence
10 phases, each a PRD + a TDD commit (data → features → env → network → training
→ backtest → SDK → terminal → GUI → docs). Status + definition-of-done: [TODO.md](TODO.md).
