"""MainWindow — thin Tkinter dashboard; renders what GuiController returns.

Coverage-omitted (pyproject ``[tool.coverage.run] omit``): a Tk window cannot
open in headless CI. All logic lives in the tested ``GuiController``; this file
only wires widgets to it. Visual verification is ``scripts/capture_gui.py``,
which launches this window and saves ``docs/assets/gui_dashboard.png``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tradedqn.gui.charts import training_figure
from tradedqn.gui.controller import GuiController
from tradedqn.gui.tcl_setup import ensure_tk_libraries


class MainWindow:
    def __init__(self, sdk) -> None:
        ensure_tk_libraries()  # point Tcl/Tk at the bundled libs before Tk() starts
        self.controller = GuiController(sdk)
        self.root = tk.Tk()
        self.root.title("TradeDQN — Dueling DQN Trading Dashboard")
        self._canvas = None
        self.status = tk.StringVar(value="Ready. Prepare data to begin.")
        self._build()

    def _build(self) -> None:
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        buttons = (
            ("Prepare data", self._prepare),
            ("Train", self._train),
            ("Backtest", self._backtest),
            ("Recommend", self._recommend),
            ("Compare arch", self._compare),
        )
        for label, handler in buttons:
            ttk.Button(bar, text=label, command=handler).pack(side=tk.LEFT, padx=4)
        self._add_inputs(bar)
        self._chart_frame = ttk.Frame(self.root)
        self._chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._progressbar = ttk.Progressbar(self.root, mode="determinate")
        self._progressbar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _add_inputs(self, bar) -> None:
        data = self.controller.sdk.cfg.data
        self._ticker = tk.StringVar(value=data.ticker)
        self._start = tk.StringVar(value=data.start)
        self._end = tk.StringVar(value=data.end)
        self._episodes = tk.IntVar(value=int(self.controller.sdk.cfg.gui.default_train_episodes))
        for label, var, width in (
            ("ticker:", self._ticker, 7), ("from:", self._start, 11), ("to:", self._end, 11),
        ):
            ttk.Label(bar, text=label).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Entry(bar, width=width, textvariable=var).pack(side=tk.LEFT)
        ttk.Label(bar, text="episodes:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(bar, from_=1, to=10000, width=6, textvariable=self._episodes).pack(side=tk.LEFT)

    def _show(self, figure) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        self._canvas = FigureCanvasTkAgg(figure, master=self._chart_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _safe(self, action) -> None:
        try:
            action()
        except (RuntimeError, ValueError) as error:
            self.status.set(f"Error: {error}")

    def _prepare(self) -> None:
        self._safe(lambda: self.status.set(
            self.controller.prepare(self._ticker.get(), self._start.get(), self._end.get())
        ))

    def _train(self) -> None:
        self._safe(self._do_train)

    def _backtest(self) -> None:
        self._safe(self._do_backtest)

    def _recommend(self) -> None:
        self._safe(self._do_recommend)

    def _compare(self) -> None:
        self._safe(self._do_compare)

    def _do_train(self) -> None:
        episodes = int(self._episodes.get())
        self._progressbar.configure(maximum=episodes, value=0)
        self.status.set(f"Training {episodes} episode(s)…")
        self.root.update_idletasks()
        status, figure = self.controller.train(episodes=episodes, on_progress=self._on_episode)
        self.status.set(status)
        self._show(figure)

    def _on_episode(self, line: str, history: list) -> None:
        self.status.set(line)
        self._progressbar.configure(value=len(history))
        self._show(training_figure(history))  # live-animate the reward + ε chart
        self.root.update_idletasks()  # repaint now; no event re-entrancy

    def _do_backtest(self) -> None:
        status, figure = self.controller.backtest()
        self.status.set(status)
        self._show(figure)

    def _do_recommend(self) -> None:
        status, figure = self.controller.recommend()
        self.status.set(status)
        self._show(figure)

    def _do_compare(self) -> None:
        episodes = int(self._episodes.get())
        self.status.set(f"Comparing Dueling vs plain DQN over {episodes} episode(s)…")
        self.root.update_idletasks()
        status, figure = self.controller.compare(episodes=episodes, on_progress=self._on_compare)
        self.status.set(status)
        self._show(figure)

    def _on_compare(self, line: str) -> None:
        self.status.set(line)
        self.root.update_idletasks()

    def run(self) -> None:
        self.root.mainloop()
