# TradeDQN — Stock-Trading Agent via a Dueling Deep Q-Network

> Bar-Ilan University · Vibe Coding Workshop · **Assignment 2**
>
> ⚠️ **Teaching tool, not investment advice.** No profitability is claimed or
> implied. Past performance does not predict future results.

> 🚧 **Scaffolding stage.** This README is a skeleton. Sections fill in as the
> phases in `docs/` complete — see
> `instructions/assignment-2/WORKING_METHODOLOGY.md` (local) for the plan.

A reinforcement-learning agent that learns a discrete **Buy / Hold / Sell**
policy on historical market data. It replaces Assignment 1's tabular Q-table
with a **Dueling DQN** (a neural network that *approximates* Q), because the
trading state space is effectively infinite.

## Objectives
_TBD — what the project demonstrates and why DQN over a Q-table._

## What's implemented
_TBD — filled per phase._

## Installation
```bash
uv sync --dev
```

## Running
_TBD — terminal menu first, then GUI. Both run through the SDK._

## Architecture
_TBD — SDK boundary; block diagram of the network (Conv1D → Value/Advantage
streams) and of the OOP class design._

## Data & the rate-limit gatekeeper (§5)
_TBD — Yahoo Finance OHLCV, ~10 years, throttled + cached to local CSV._

## Results & analysis
_TBD — training curves, held-out backtest equity curve / P&L, conclusions._

## Cost of AI-assisted development (§11)
_TBD._

## Quality bar
TDD · 85%+ coverage · ≤150 code lines/file · zero ruff violations · secret-scan
· uv-only. Enforced by pre-commit + CI.

## Tests
```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## License
MIT — see [LICENSE](LICENSE).
