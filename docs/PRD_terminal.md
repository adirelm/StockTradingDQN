# PRD — Phase 8: Terminal interface (menu over the SDK)

**Phase:** 8 of 10 · **Status:** approved-to-build · Covers `REQUIREMENTS.md`
**B21** (and B23 pipeline at the UI). The lecturer stressed building the
terminal UI **first**: an AI agent can drive and self-test it without a mouse.

## Goal
A thin, presentation-only menu that calls **only** the SDK
(`Prepare Data → Train → Backtest → Predict`, plus save/load), robust to misuse.

## Module
- `cli/menu.py` — `TerminalApp(sdk, input_fn=input, output_fn=print)`:
  - `run()` loops: print menu → read choice → dispatch → repeat until Quit.
  - `input_fn`/`output_fn` are injected so tests drive it with no real stdin
    and capture output (also makes it agent-drivable).
  - Each handler is wrapped: an SDK error (e.g. train before prepare) is caught
    and shown as a message — the menu never crashes.
- `main.py` (repo root) — entry point: `TerminalApp(TradingSDK()).run()`.

## Menu
```
1 Prepare data     2 Train     3 Backtest     4 Recommend next action
5 Save brain       6 Load brain               0 Quit
```

## Acceptance criteria (tests assert; FakeSDK stub, no network/training)
- `run()` with a scripted input sequence dispatches to the right SDK calls in
  order (prepare → train → backtest → recommend → save → load → quit).
- The menu text lists all options; an unknown choice prints a message and loops.
- Quit ("0") exits cleanly.
- An SDK error during a handler is caught and surfaced (no traceback escapes).
- Output for backtest/recommend includes the key numbers (total return, action).

## Gates
≤150 code lines/file · TDD · coverage ≥85% · ruff clean. (`main.py` is the
non-tested entry shim — kept to 3 lines.)
