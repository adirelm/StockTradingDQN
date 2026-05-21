# PRD — TradeDQN (umbrella product requirements)

The project-wide requirements document (§2.2). Per-mechanism detail lives in the
layered `docs/PRD_<phase>.md` files, linked below.

> ⚠️ Teaching tool, **not investment advice**. Past performance ≠ future results.

## 1. Objective & scope
Demonstrate the progression **finite Q-table → Bellman update → neural
approximation (DQN) → a working DQN-stock system**: an RL agent that learns a
discrete **Buy / Hold / Sell** policy on historical market data using a **Dueling
Deep Q-Network**, exposed through one SDK with a terminal and a GUI.

**In scope:** data ingestion (Yahoo Finance + gatekeeper + cache), feature
engineering, a Gym-style trading environment, the Dueling DQN + replay + target
network, training, backtesting vs Buy & Hold, single-step inference, terminal +
GUI, full docs.
**Out of scope:** live/broker trading, multi-asset portfolios, intraday data,
profitability. (The agent is *not* expected to beat the market.)

## 2. Users & user stories
- *Student/learner*: "Prepare data → train → backtest → see whether the policy
  beats Buy & Hold, and read an honest analysis of why/why not."
- *Grader*: "Run it from a clean checkout; read the README + per-phase PRDs;
  verify the gates and the honest results."
- *AI agent (Vibe Coding)*: "Drive the whole pipeline through the terminal menu
  / SDK without a GUI."

## 3. Functional requirements (→ per-phase PRDs)
| Area | Requirement | PRD |
|---|---|---|
| Data | OHLCV from Yahoo, §5 rate-limit gatekeeper, CSV cache | [PRD_data](PRD_data.md) |
| Features | 8 market indicators, normalize fit-on-train, chrono split | [PRD_features](PRD_features.md) |
| Env | 30×10 state, Sell/Hold/Buy, reward `ΔV−C−S+λ·Sharpe` | [PRD_env](PRD_env.md) |
| Network | Dueling Conv1D DQN (V + A heads) | [PRD_network](PRD_network.md) |
| Training | replay buffer, target net, ε-greedy, MSE Bellman | [PRD_training](PRD_training.md) |
| Backtest | equity vs Buy&Hold + metrics; inference | [PRD_backtest](PRD_backtest.md) |
| SDK | single facade; UIs depend only on it | [PRD_sdk](PRD_sdk.md) |
| Terminal | menu over the SDK (built first) | [PRD_terminal](PRD_terminal.md) |
| GUI | Tkinter + matplotlib dashboard | [PRD_gui](PRD_gui.md) |

## 4. Non-functional requirements
TDD with ≥85% coverage (held at 100%); ≤150 code lines/file; zero ruff
violations; no hardcoded values (config only); secret-scan clean; uv-only;
deterministic seeds; cross-platform (CPU/MPS), `uv`-locked install.

## 5. Acceptance criteria / KPIs
- Clean-checkout `uv sync && uv run pytest` is green (≥85% coverage).
- `uv run main.py` (terminal) and `uv run main.py gui` both run through the SDK.
- `uv run python scripts/generate_results.py` reproduces the README's real
  backtest (equity curve vs Buy & Hold + metrics).
- Every requirement in `instructions/assignment-2/REQUIREMENTS.md` has evidence.
- **Honesty KPI:** the held-out result is reported as-is (the agent currently
  *underperforms* Buy & Hold — see README Conclusions).

## 6. Constraints & assumptions
Single ticker, daily bars, ~10 years; single-threaded (see PLAN §concurrency);
Yahoo rate limits → cache-first; "past ≠ future" — no profitability claim.

## 7. Architecture, plan & tasks
Design + diagrams + ADRs + concurrency: [PLAN.md](PLAN.md). Phased task list with
definition-of-done: [TODO.md](TODO.md). AI workflow + decisions: [PROMPTS.md](PROMPTS.md).
