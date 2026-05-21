"""MainWindow — thin Tkinter dashboard; renders what GuiController returns.

Coverage-omitted (pyproject ``[tool.coverage.run] omit``): a Tk window cannot
open in headless CI. All logic lives in the tested ``GuiController``; this file
only wires widgets to it. Visual verification is the screenshot-regression
methodology phase.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tradedqn.gui.controller import GuiController


class MainWindow:
    def __init__(self, sdk) -> None:
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
        )
        for label, handler in buttons:
            ttk.Button(bar, text=label, command=handler).pack(side=tk.LEFT, padx=4)
        self._chart_frame = ttk.Frame(self.root)
        self._chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6).pack(
            side=tk.BOTTOM, fill=tk.X
        )

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
        self._safe(lambda: self.status.set(self.controller.prepare()))

    def _train(self) -> None:
        self._safe(self._do_train)

    def _backtest(self) -> None:
        self._safe(self._do_backtest)

    def _recommend(self) -> None:
        self._safe(lambda: self.status.set(self.controller.recommend()))

    def _do_train(self) -> None:
        status, figure = self.controller.train()
        self.status.set(status)
        self._show(figure)

    def _do_backtest(self) -> None:
        status, figure = self.controller.backtest()
        self.status.set(status)
        self._show(figure)

    def run(self) -> None:
        self.root.mainloop()
