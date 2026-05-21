"""Tests for the GUI chart builders (B22 — pure matplotlib figures, headless)."""

from matplotlib.figure import Figure

from tradedqn.gui.charts import equity_figure, training_figure


class TestEquityFigure:
    def test_returns_figure_with_two_labelled_lines(self):
        fig = equity_figure([1000, 1100, 1050], [1000, 1080, 1120])
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        labels = [line.get_label() for line in ax.get_lines()]
        assert labels == ["DQN policy", "Buy & Hold"]

    def test_title_mentions_buy_and_hold(self):
        fig = equity_figure([1, 2], [1, 2])
        assert "Buy & Hold" in fig.axes[0].get_title()


class TestTrainingFigure:
    def test_plots_episode_reward(self):
        history = [{"reward": 1.0}, {"reward": 2.5}, {"reward": 2.0}]
        fig = training_figure(history)
        assert isinstance(fig, Figure)
        line = fig.axes[0].get_lines()[0]
        assert list(line.get_ydata()) == [1.0, 2.5, 2.0]
