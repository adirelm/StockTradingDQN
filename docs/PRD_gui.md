# PRD — Phase 9: GUI dashboard (Tkinter + matplotlib over the SDK)

**Phase:** 9 of 10 · **Status:** approved-to-build (ADR: Tkinter+matplotlib) ·
Covers `REQUIREMENTS.md` **B22** (and feeds the README block diagrams B24/B25).

## Goal
A desktop dashboard over the **same SDK** the terminal uses: buttons for
Prepare / Train / Backtest / Recommend / **Compare arch** (Dueling-vs-plain DQN),
ticker / date-range / episode-count input widgets, an embedded chart, a
**live-animating** training view (reward + ε curve refreshed per episode with a
**progress bar**), and a status line — polished and presentation-ready on top of
the already-working terminal app.

## Shipped widgets (as built — `gui/app.py`, `gui/charts.py`)
- **Toolbar** — five `ttk.Button`s in pipeline order: Prepare data, Train,
  Backtest, Recommend, **Compare arch**.
- **Inputs** (`_add_inputs`) — `ttk.Entry` for ticker / from / to (seeded from
  `cfg.data`) and a `ttk.Spinbox` for episode count (seeded from
  `cfg.gui.default_train_episodes`, default 20).
- **Live training** — `_do_train` sets the `ttk.Progressbar` maximum to the
  episode count; the `_on_episode` callback updates the status line, advances the
  progress bar, and re-renders `training_figure(history)` each episode (reward +
  ε on a twin axis, mean MSE loss below).
- **Backtest** — `backtest_figure`: two panels — price with **▲ buy / ▼ sell**
  trade markers, and the equity curve vs Buy & Hold.
- **Recommend** — `q_value_figure`: a **Q-value bar chart** over the config
  action labels, with the chosen action highlighted.
- **Compare arch** — `comparison_figure`: overlaid Dueling-vs-plain reward curves.

## Decision (ADR)
Tkinter (stdlib) + matplotlib (already a dependency) embedded via
`FigureCanvasTkAgg`. No new dependency; mirrors the deck's `gui/` structure.

## Modules + testability split
- `gui/charts.py` — **pure** figure builders returning `matplotlib.figure.Figure`
  (no pyplot, no display): `equity_figure`, `backtest_figure` (price + ▲/▼ trade
  markers + equity vs Buy & Hold), `training_figure` (reward + ε + mean loss),
  `comparison_figure` (Dueling vs plain), `q_value_figure` (action Q-value bars).
  Fully unit-tested headless.
- `gui/controller.py` — `GuiController(sdk)`: action methods that call the SDK
  and return `(status_text, figure)` (or status text), including `compare()` and
  an `on_progress` callback that streams per-episode records for the live chart.
  No Tk → fully unit-tested.
- `gui/app.py` — `MainWindow(sdk)`: thin Tk wiring (toolbar buttons → controller,
  ticker/date/episode inputs, embedded canvas, progress bar, status label,
  try/except around actions). **Coverage-omitted** — a Tk window can't open
  headless in CI; visual verification is the screenshot-regression methodology
  phase. Kept minimal.
- `gui/tcl_setup.py` — `ensure_tk_libraries()` auto-locates the interpreter's
  bundled Tcl/Tk dirs (setting `TCL_LIBRARY`/`TK_LIBRARY` when unset) so
  `uv run main.py gui` launches under the uv-managed Python; called once before
  the Tk root is created.
- `main.py` — `uv run main.py gui` launches the dashboard; default = terminal.

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

## Visual reference & UX
- **Screenshot:** the committed dashboard image is `docs/assets/gui_dashboard.png`
  (captured by `scripts/capture_gui.py`).
- **Usability:** the Nielsen 10-heuristics walkthrough and accessibility notes
  (keyboard-operable, labelled controls) live in the README "User interface &
  UX (§10)" section — not duplicated here.
