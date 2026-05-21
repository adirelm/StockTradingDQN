"""Chart builders — pure matplotlib figures (no pyplot, no display).

Returning a ``Figure`` directly keeps these testable headless and lets the Tk
window embed them via ``FigureCanvasTkAgg``.
"""

from __future__ import annotations

from matplotlib.figure import Figure

# UI-styling literals (size/dpi) are local presentation choices, not config.
_FIGSIZE = (6.0, 3.2)
_DPI = 100


def equity_figure(equity, benchmark) -> Figure:
    """Strategy equity vs the Buy & Hold benchmark."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    ax.plot(equity, label="DQN policy", color="#1f77b4")
    ax.plot(benchmark, label="Buy & Hold", color="#888888", linestyle="--")
    ax.set_title("Equity curve vs Buy & Hold")
    ax.set_xlabel("trading step")
    ax.set_ylabel("portfolio value")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def training_figure(history) -> Figure:
    """Per-episode training reward curve."""
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot(111)
    rewards = [record["reward"] for record in history]
    ax.plot(rewards, label="episode reward", color="#2ca02c")
    ax.set_title("Training reward per episode")
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
