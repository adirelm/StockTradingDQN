# PROMPTS — how this project was built with AI

TradeDQN was built with **Claude Code** using a **Vibe Coding** methodology:
the human acts as architect (decides scope, architecture, reward design,
acceptance criteria, the self-score), the AI implements against an approved
spec. This log shows the workflow, the decisions, and where human judgment
overrode the AI's first instinct — not just "the AI wrote the code".

## The workflow: PRD → PLAN → TODO → Validate → Execute

Every layer started from a written **PRD** in [`../docs/`](.) that the human
approved *before* any code was generated, then was built **test-first** (TDD,
RED→GREEN→REFACTOR) under the gates in `CLAUDE.md` (≤150 code lines/file, 85%+
coverage — we held 100%, zero ruff violations, secret-scan, uv-only).

The build was deliberately split into **10 sequential phases**, each its own
PRD + commit, so the git history shows the development *arc* (the lecturer
grades this) rather than one bulk dump:

| Phase | PRD | What landed |
|------|-----|-------------|
| 1 | PRD_data | `DataClient` + §5 rate-limit gatekeeper + CSV cache |
| 2 | PRD_features | 8 market indicators + normalization (fit-on-train) + chronological split |
| 3 | PRD_env | `TradingEnvironment` (30×10 state, Buy/Hold/Sell, reward) |
| 4 | PRD_network | Dueling Conv1D DQN |
| 5 | PRD_training | replay buffer + target net + DQN agent + training loop |
| 6 | PRD_backtest | backtest metrics + inference |
| 7 | PRD_sdk | `TradingSDK` facade (the §4 mandate) |
| 8 | PRD_terminal | terminal menu (built before the GUI, lecturer's order) |
| 9 | PRD_gui | Tkinter + matplotlib dashboard |
| 10 | PRD_docs | this README/PROMPTS/cost analysis + diagrams |

## Architect decisions the AI did not get to make (ADRs)

These were **human-decided** and recorded, because they change behaviour:

- **Config format** ([ADR-001](ADR-001-config-format.md)): YAML over the
  lecturer's `setup.json` — YAML carries an inline "why" per hyperparameter —
  while keeping his exact parameter *names* for cross-mapping.
- **Position model** (PRD_env D1): all-in / all-out (the deck's simple "have
  stock or not"), not fractional sizing.
- **Reward** (PRD_env D3): the full deck formula
  `rₜ = ΔVₜ − Cₜ − Sₜ + λ·Sharpeₜ`, not raw profit — so over-trading, costs,
  slippage and risk are penalised.
- **GUI tech** (PRD_gui): Tkinter + matplotlib (no new dependency) over
  Streamlit, to keep the strict SDK-facade and stay unit-testable.

## Where human judgment corrected the AI

- The AI's first instinct was to precompute all 10 state features. The human
  caught that `position` and `cash_exposure` are **path-dependent** on the
  agent's own trades — precomputing them is meaningless and leaks. They are now
  injected by the env at runtime; only the 8 market features are precomputed.
- Normalization is **fit on train only** and clips val/test — the AI's naive
  version would have fit on the whole series (look-ahead leakage), the cardinal
  sin of financial ML. A test (`clip-future-highs`) now guards it.
- Checkpoints load with `weights_only=True` (no arbitrary-code execution),
  carried over as a deliberate §7 security rule.

## Verbatim orchestrating prompt (per phase)

> *You are implementing Phase N of TradeDQN against `docs/PRD_<phase>.md`,
> which the human approved. Work test-first (RED→GREEN→REFACTOR). Obey
> `CLAUDE.md`: ≤150 code lines/file, ≥85% coverage, zero ruff, no hardcoded
> values (config only), SDK is the single business-logic entry point. Stop and
> surface any decision that changes a human-decided concern (reward, API shape,
> architecture) before landing it.*

## Pre-submission review

Before submission we run a multi-pass, one-section-per-step review against the
course guidelines: walk every requirement, one at a time, and only check it off
with concrete evidence (a file path, a test, a screenshot). The review notes and
the requirement register are kept as **local working files** (not part of the
submission) — the instrument that stops a requirement from being missed.
