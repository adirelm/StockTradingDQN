# PRD — Phase 9: GUI dashboard (Tkinter + matplotlib over the SDK)

**Phase:** 9 of 10 · **Status:** approved-to-build (ADR: Tkinter+matplotlib) ·
Covers `REQUIREMENTS.md` **B22** (and feeds the README block diagrams B24/B25).

## Goal
A desktop dashboard over the **same SDK** the terminal uses: buttons for
Prepare / Train / Backtest / Recommend, an embedded chart (equity vs Buy&Hold,
or the training-reward curve), and a status line — polished and presentation-ready on top of the already-working terminal app.

## Decision (ADR)
Tkinter (stdlib) + matplotlib (already a dependency) embedded via
`FigureCanvasTkAgg`. No new dependency; mirrors the deck's `gui/` structure.

## Modules + testability split
- `gui/charts.py` — **pure** figure builders returning `matplotlib.figure.Figure`
  (no pyplot, no display): `equity_figure(equity, benchmark)`,
  `training_figure(history)`. Fully unit-tested headless.
- `gui/controller.py` — `GuiController(sdk)`: action methods that call the SDK
  and return `(status_text, figure)` (or status text). No Tk → fully unit-tested.
- `gui/app.py` — `MainWindow(sdk)`: thin Tk wiring (buttons → controller, canvas,
  status label, try/except around actions). **Coverage-omitted** — a Tk window
  can't open headless in CI; visual verification is the screenshot-regression
  methodology phase. Kept minimal.
- `main.py` — `python main.py gui` launches the dashboard; default = terminal.

## Acceptance criteria (tests assert; FakeSDK / Agg figures, headless)
- charts: `equity_figure` has 2 lines labelled DQN + Buy&Hold; `training_figure`
  plots one reward line; both return a `Figure` with one axes.
- controller: `prepare()` status contains the split sizes; `train()` returns
  (status, Figure); `backtest()` returns (status, Figure) and caches the result;
  `recommend()` status contains the action; an SDK error propagates as a clear
  message via the controller's safe wrapper.

## Gates
≤150 code lines/file · TDD (controller+charts) · coverage ≥85% · ruff clean.
`gui/app.py` omitted from coverage (documented, headless-untestable Tk).
