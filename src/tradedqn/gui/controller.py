"""GuiController — the GUI's logic, with no Tk dependency (so it unit-tests).

Each method calls the SDK and returns a status string (and a figure for the
chart actions). The Tk window just renders what these return.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from tradedqn.format import backtest_line, recommendation_line
from tradedqn.gui.charts import (
    backtest_figure,
    comparison_figure,
    q_value_figure,
    training_figure,
)


def _episode_line(record: dict) -> str:
    """Format one episode's live progress line (number, ε, reward)."""
    return (
        f"Episode {record['episode'] + 1}: "
        f"ε={record['epsilon']:.3f}, reward={record['reward']:.2f}"
    )


class GuiController:
    """GUI logic with no Tk dependency: each method calls the SDK and returns text (+a figure)."""

    def __init__(self, sdk) -> None:
        self.sdk = sdk
        self.last_backtest: dict | None = None

    def prepare(self, ticker=None, start=None, end=None) -> str:
        """Prepare data (optional ticker/date overrides); return a status line."""
        sizes = self.sdk.prepare_data(ticker=ticker, start=start, end=end)
        return f"Prepared splits: {sizes}"

    def train(self, episodes=None, on_progress=None) -> tuple[str, Figure]:
        """Train (optionally for ``episodes``) and stream live progress.

        ``on_progress(line, history)`` fires as each episode finishes — the line
        for the status bar, the running history for a live-animating chart — so
        the window updates continuously instead of looking frozen.
        """
        history: list[dict] = []

        def relay(record: dict) -> None:
            """Track the record and relay a formatted progress line."""
            history.append(record)
            if on_progress is not None:
                on_progress(_episode_line(record), list(history))

        self.sdk.train(episodes=episodes, on_episode=relay)
        last = history[-1]
        return (
            f"Trained {last['episode'] + 1} episode(s); ε={last['epsilon']:.3f}",
            training_figure(history),
        )

    def backtest(self) -> tuple[str, Figure]:
        """Run the backtest; cache it and return (status line, two-panel figure)."""
        result = self.sdk.backtest()
        self.last_backtest = result
        return f"Backtest — {backtest_line(result)}", backtest_figure(result)

    def recommend(self) -> tuple[str, Figure]:
        """Recommend an action; return (status line, Q-value bar figure)."""
        rec = self.sdk.recommend()
        labels = [name.title() for name in rec["names"]]  # config-ordered action labels
        return recommendation_line(rec), q_value_figure(rec["q_values"], rec["action_index"], labels)

    def compare(self, episodes=None, on_progress=None) -> tuple[str, Figure]:
        """Train Dueling vs plain DQN and chart both reward curves (§9 ablation)."""

        def relay(name: str, record: dict) -> None:
            """Relay each architecture's per-episode progress line."""
            if on_progress is not None:
                on_progress(f"{name} — {_episode_line(record).lower()}")

        histories = self.sdk.compare(episodes=episodes, on_episode=relay)
        finals = ", ".join(f"{name}: {h[-1]['reward']:.1f}" for name, h in histories.items())
        return f"Final reward — {finals}", comparison_figure(histories)
