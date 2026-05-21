"""Chart builders — pure matplotlib figures (no pyplot, no display).

Returning a ``Figure`` directly keeps these testable headless and lets the Tk
window embed them via ``FigureCanvasTkAgg``.
"""

from __future__ import annotations

from matplotlib.figure import Figure

# UI-styling literals (size/dpi) are local presentation choices, not config.
_FIGSIZE = (6.0, 3.2)
_DPI = 160  # high-resolution figures for the README / docs (§9.3)
_ACTIONS = ("Sell", "Hold", "Buy")


def _plot_equity(ax, equity, benchmark) -> None:
    """Draw the DQN-vs-Buy&Hold equity lines on ``ax``."""
    ax.plot(equity, label="DQN policy", color="#1f77b4")
    ax.plot(benchmark, label="Buy & Hold", color="#888888", linestyle="--")
    ax.set_title("Equity curve vs Buy & Hold")
    ax.set_xlabel("trading step")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best")


def _plot_prices(ax, prices, markers) -> None:
    """Draw the price line with ▲buy / ▼sell trade markers on ``ax``."""
    ax.plot(prices, color="#444444", linewidth=1.0, label="price")
    for side, sym, color in (("buy", "^", "#2ca02c"), ("sell", "v", "#d62728")):
        pts = [m for m in markers if m["action"] == side]
        if pts:
            ax.scatter([m["step"] for m in pts], [m["price"] for m in pts],
                       marker=sym, color=color, s=60, zorder=3, label=side)
    ax.set_title("Price with agent trades")
    ax.set_ylabel("price")
    ax.legend(loc="best")


def equity_figure(equity, benchmark) -> Figure:
    """Strategy equity vs the Buy & Hold benchmark."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    _plot_equity(fig.add_subplot(111), equity, benchmark)
    fig.tight_layout()
    return fig


def backtest_figure(result) -> Figure:
    """Two stacked panels: price + buy/sell markers, and equity vs Buy & Hold."""
    fig = Figure(figsize=(6.0, 4.8), dpi=_DPI)
    _plot_prices(fig.add_subplot(211), result["price_curve"], result["trade_markers"])
    _plot_equity(fig.add_subplot(212), result["equity_curve"], result["benchmark_curve"])
    fig.tight_layout()
    return fig


def training_figure(history) -> Figure:
    """Per-episode training reward (left axis) and ε decay (right axis)."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    ax.plot([r["reward"] for r in history], label="episode reward", color="#2ca02c")
    ax.set_title("Training progress")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    if history and "epsilon" in history[0]:
        eps = ax.twinx()
        eps.plot([r["epsilon"] for r in history], label="ε", color="#ff7f0e", linestyle=":")
        eps.set_ylabel("ε (exploration)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def comparison_figure(histories) -> Figure:
    """Overlay each architecture's per-episode reward (Dueling vs plain DQN)."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    for name, history in histories.items():
        ax.plot([r["reward"] for r in history], label=name)
    ax.set_title("Dueling vs plain DQN — training reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def q_value_figure(q_values, action_index) -> Figure:
    """Bar chart of the three action Q-values, the chosen one highlighted."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    colors = ["#bbbbbb"] * len(_ACTIONS)
    colors[action_index] = "#1f77b4"
    ax.bar(_ACTIONS, q_values, color=colors)
    ax.set_title(f"Q-values → {_ACTIONS[action_index]}")
    ax.set_ylabel("Q(s, a)")
    fig.tight_layout()
    return fig
