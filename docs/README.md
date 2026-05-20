# docs/

This folder holds the **layered PRDs**, plans, TODO lists, and the prompt log —
the documentation trail the lecturer grades (he asked explicitly to see work in
*layers*, not one PRD).

Build order (see `instructions/assignment-2/WORKING_METHODOLOGY.md`):

| Phase | PRD | Delivers |
|------|-----|----------|
| 1 | `PRD_data.md` | yfinance fetch + §5 rate-limit gatekeeper + CSV cache |
| 2 | `PRD_features.md` | technical indicators + normalisation + train/val/test split |
| 3 | `PRD_env.md` | Gym-style trading env: 30×10 state, Buy/Hold/Sell, reward |
| 4 | `PRD_network.md` | Dueling DQN (Conv1D 32→64 → Value + Advantage streams) |
| 5 | `PRD_training.md` | replay buffer, ε-schedule, Bellman target, MSE loss, loop |
| 6 | `PRD_backtest.md` | held-out evaluation, equity curve / P&L, metrics |
| 7 | `PRD_sdk.md` | SDK facade (single business-logic entry point) |
| 8 | `PRD_terminal.md` | terminal menu over the SDK (built first — agent-drivable) |
| 9 | `PRD_gui.md` | GUI + training interface over the SDK |
| 10 | `PRD_docs.md` | README, block diagrams, results analysis, cost + ISO mapping |

Each PRD is approved by the human (architect) **before** code is generated
against it — `CLAUDE.md` §1.4 contract.

Also planned here: `PROMPTS.md` (verbatim prompt log), `COST_ANALYSIS.md`
(§11 token/cost), and architecture / OOP block diagrams referenced by the README.
