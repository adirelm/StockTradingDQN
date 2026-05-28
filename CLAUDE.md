# CLAUDE.md — Global Coding Standards for TradeDQN

## Project Context

TradeDQN is a reinforcement-learning project that trains a **stock-trading
agent** with a **Dueling Deep Q-Network (DQN)**. The agent observes a window
of recent market features and learns a discrete policy — **Buy / Hold / Sell**
— that maximises cumulative, risk- and cost-adjusted return.

It is Assignment 2 of the Bar-Ilan University Vibe Coding Workshop. It is a
**fresh repository**, but it deliberately reuses the engineering scaffolding,
quality gates, and review methodology proven on Assignment 1 (DroneRL).

This is a teaching tool, **not investment advice**. No claim of profitability
is made or implied.

> **Source of truth for requirements.** Two authoritative sources: (1) the
> course software-engineering guidelines booklet (§1–§20, generic to every
> assignment), and (2) this assignment's brief + lecture material. When in doubt,
> the booklet + the lecturer's words win over anything in this file.

## How We Work (non-negotiable process)

We walk **every requirement, slowly, one at a time**. The guiding lesson:
*missing* a requirement the rest of the class addressed costs more than any
single quality slip. So:

- Nothing is "done" until its requirement row is checked **with
  evidence** (a file path, a test name, a screenshot).
- Human decides scope / acceptance criteria / architecture; AI implements
  against an approved PRD. (Same §1.4 architect-vs-implementer contract as
  Assignment 1 — see the table below.)
- PRD → PLAN → TODO → Validate → Execute. Layered PRDs, not one dump.

## Human ↔ AI Responsibility Contract (§1.4)

| Concern | Human-decided (non-delegable) | AI-delegated |
|---|---|---|
| Requirements (PRD, scope, success criteria, KPIs) | ✅ | — |
| Architecture (network design, layer boundaries, SDK API shape) | ✅ | — |
| Test acceptance criteria + the assertions that must hold | ✅ | — |
| Reward-function design + financial-metric choices | ✅ | — |
| Final code-review sign-off + commit message intent | ✅ | — |
| Self-score / grade claim against the rubric | ✅ | — |
| Code generation against an approved spec | — | ✅ |
| Refactoring within an existing public API | — | ✅ |
| Test scaffolding + boilerplate from a written spec | — | ✅ |
| Docstring drafts (human edits before commit) | — | ✅ |
| Lint / format auto-fixes | — | ✅ |

**Operating rule.** If an AI change would alter a human-decided row (new SDK
method, changed test assertion, weakened gate, new architecture), the human
signs off on the PRD/PLAN edit *first*, then the AI executes against it.

## Hard Constraints (Apply to ALL Files)

### 1. File Size Limit — 150 Lines Maximum
Every Python file (.py) must not exceed 150 **code** lines (blank lines and
`#` comment lines do not count). If a file approaches the limit, split it.
Enforced by `scripts/check_file_sizes.sh` (pre-commit + CI).

### 2. Test-Driven Development (TDD)
Write tests BEFORE implementation. RED → GREEN → REFACTOR.
85%+ coverage gate. Run: `uv run pytest tests/ --cov=src/tradedqn --cov-report=term-missing`

### 3. Object-Oriented Programming (OOP)
Inheritance over duplication (e.g. `RolloutService → TrainingService / BacktestService`; `DuelingDQN(nn.Module)` with the dueling head as a config flag, not a subclass).
The **SDK is the single entry point** for all business logic. UIs (terminal +
GUI) call the SDK; they never touch training / data / network code directly.

### 4. No Hardcoded Values
ALL algorithm-relevant parameters live in `config/config.yaml` and are read
via the config loader: network dims, learning rate, gamma, epsilon schedule,
window size, feature list, replay-buffer size, transaction-cost/slippage
coefficients, ticker symbols, date ranges, seeds. Local UI-styling literals
(pixel sizes, matplotlib `dpi`/`alpha`) stay in their rendering modules.
Test: *"would a grader/contributor/future-me want to change this without
editing source?"* Yes → config. No → keep local.

### 5. No Code Duplication (DRY)
If a pattern appears twice, extract a utility or base-class method.

### 6. Linting — Zero Ruff Violations
`uv run ruff check src/ tests/ scripts/ main.py` → zero errors before commit.

### 7. Package Manager — UV Only
`uv` exclusively. `uv sync --dev` to install, `uv run …` to run.

## Domain Requirements (this assignment)

### Data (§5 — API Gatekeeper is now load-bearing)
- Source: **Yahoo Finance** via `yfinance`. OHLCV, multi-year (this build: ~3 years, 2020–2023 — the binding §4 window).
- **Rate-limit gatekeeper is mandatory**: throttle / cache so we never hammer
  the API. Persist raw pulls to a local parquet cache (+ CSV fallback) for reproducibility and offline runs.

### Features & State
- Compute technical indicators (moving averages, volatility, momentum, volume
  change, ratio-to-MA, normalised price, …). **Normalise to ~[0, 1]**.
- State = a window of **30 days × ~10 features** (matrix), fed to the network.
- Train / validation / test split. Sliding-window (or random-window) sampling.

### Network — Dueling DQN
- Input 30×10 → **Conv1D over the time axis only** (the feature axis has no
  ordering, so never Conv2D across it) → 32ch → 64ch → Flatten → FC.
- **Two streams**: Value V(s) (scalar) + Advantage A(s,a) (vector of 3) →
  combined into Q(s, Buy/Hold/Sell).
- Loss = MSE between the Bellman target and the predicted Q.

### Reward
- Risk- and cost-adjusted return — penalise excessive trading, transaction
  costs, slippage, excess risk. Not raw profit. Discount with gamma.

### Backtesting
- Train on history; evaluate on a held-out slice; report the equity curve /
  P&L over that period. Always state the "past ≠ future" caveat honestly.

## Interfaces (§4 / §10)
- **Terminal menu first** (agent-friendly: an AI agent can drive + self-test it
  without a mouse), then a **GUI** on top. Both go through the SDK only.

## Deliverables (§2 / §9)
- Working GitHub project. Rich README: objective, install, usage, screenshots,
  charts, **architecture block diagram**, **OOP/class block diagram**,
  gatekeeper description, results analysis + conclusions.
- `docs/` with **layered PRDs**, PLANs, TODO lists, and a PROMPTS log.
- Tests proving the network learns (backtest evidence).

## Version Control
- Fresh repository. Default branch: `main`. Semantic version in `pyproject.toml`.
- Commit history must show the development arc (many meaningful commits), not
  one bulk dump — the lecturer grades this explicitly.

## Pre-Submission Review Methodology
Before any submission (or when asked "is this ready?"), run a structured
self-critique pass: walk **every** brief/guideline requirement one at a time,
map each to concrete evidence (file/test/artifact) or flag it as a gap, then
re-run all gates (ruff · tests + coverage · ≤150 lines · secret-scan) and verify
a fresh clone reproduces the headline before claiming done.
