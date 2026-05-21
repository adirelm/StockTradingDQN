"""GuiController — the GUI's logic, with no Tk dependency (so it unit-tests).

Each method calls the SDK and returns a status string (and a figure for the
chart actions). The Tk window just renders what these return.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from tradedqn.gui.charts import equity_figure, training_figure


class GuiController:
    def __init__(self, sdk) -> None:
        self.sdk = sdk
        self.last_backtest: dict | None = None

    def prepare(self) -> str:
        return f"Prepared splits: {self.sdk.prepare_data()}"

    def train(self) -> tuple[str, Figure]:
        history = self.sdk.train()
        last = history[-1]
        status = f"Trained {last['episode'] + 1} episode(s); ε={last['epsilon']:.3f}"
        return status, training_figure(history)

    def backtest(self) -> tuple[str, Figure]:
        result = self.sdk.backtest()
        self.last_backtest = result
        status = (
            f"Return {result['total_return']:.2%} vs Buy&Hold "
            f"{result['benchmark_return']:.2%}  ·  Sharpe {result['sharpe_ratio']:.2f}  ·  "
            f"max DD {result['max_drawdown']:.2%}  ·  trades {result['num_trades']}"
        )
        return status, equity_figure(result["equity_curve"], result["benchmark_curve"])

    def recommend(self) -> str:
        rec = self.sdk.recommend()
        q = ", ".join(f"{v:.3f}" for v in rec["q_values"])
        return f"Recommended action: {rec['action'].upper()}  (Q = [{q}])"
