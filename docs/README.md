# docs/

This folder holds the **layered PRDs**, plans, TODO lists, and the prompt log —
the documentation trail the lecturer grades (he asked explicitly to see work in
*layers*, not one PRD).

Build order (one PRD + one TDD commit per phase):

| Phase | PRD | Delivers |
|------|-----|----------|
| 1 | `PRD_data.md` | yfinance fetch + §5 rate-limit gatekeeper + parquet cache |
| 2 | `PRD_features.md` | technical indicators + normalisation + train/val/test split |
| 3 | `PRD_env.md` | Gym-style trading env: 30×10 state, Buy/Hold/Sell, reward |
| 4 | `PRD_network.md` | Dueling DQN (Conv1D 32→64 → Value + Advantage streams) |
| 5 | `PRD_training.md` | replay buffer, ε-schedule, Bellman target, MSE loss, loop |
| 6 | `PRD_backtest.md` | held-out evaluation, equity curve / P&L, metrics |
| 7 | `PRD_sdk.md` | SDK facade (single business-logic entry point) |
| 8 | `PRD_terminal.md` | terminal menu over the SDK (built first — agent-drivable) |
| 9 | `PRD_gui.md` | GUI + training interface over the SDK |
| 10 | the root `README.md` | block diagrams, results analysis, cost + ISO mapping (the deliverable itself) |

Each PRD is approved by the human (architect) **before** code is generated
against it — `CLAUDE.md` §1.4 contract.

Umbrella documents in this folder: [`PRD.md`](PRD.md) (project-wide requirements,
§2.2.a), [`PLAN.md`](PLAN.md) (architecture + ADRs + concurrency, §2.2.b),
[`TODO.md`](TODO.md) (phased tasks + DoD, §2.2.c), [`PROMPTS.md`](PROMPTS.md)
(prompt log), [`COST_ANALYSIS.md`](COST_ANALYSIS.md) (§11 token/cost), and
[`ADR-001-config-format.md`](ADR-001-config-format.md) (YAML config decision).

Reading order for a grader: root [`README.md`](../README.md) → [`PRD.md`](PRD.md)
→ the layered `PRD_*.md` (phases 1–9 above) → [`PROMPTS.md`](PROMPTS.md) →
[`COST_ANALYSIS.md`](COST_ANALYSIS.md).
