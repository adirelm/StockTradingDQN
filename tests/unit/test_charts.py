"""Tests for the GUI chart builders (B22 — pure matplotlib figures, headless)."""

from matplotlib.figure import Figure

from tradedqn.gui.charts import (
    backtest_figure,
    comparison_figure,
    equity_figure,
    q_value_figure,
    training_figure,
)


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

    def test_adds_epsilon_twin_axis_when_present(self):
        history = [{"reward": 1.0, "epsilon": 0.9}, {"reward": 2.0, "epsilon": 0.7}]
        fig = training_figure(history)
        assert len(fig.axes) == 3  # reward axis + ε twin axis + loss subplot
        assert list(fig.axes[1].get_lines()[0].get_ydata()) == [0.9, 0.7]

    def test_plots_loss_subplot_skipping_warmup_none(self):
        history = [{"reward": 1.0, "mean_loss": None}, {"reward": 2.0, "mean_loss": 0.5},
                   {"reward": 1.5, "mean_loss": 0.3}]
        fig = training_figure(history)
        loss_ax = fig.axes[-1]  # bottom subplot
        assert list(loss_ax.get_lines()[0].get_ydata()) == [0.5, 0.3]  # None warmup skipped
        assert list(loss_ax.get_lines()[0].get_xdata()) == [1, 2]      # at their episode indices


class TestBacktestFigure:
    RESULT = {
        "price_curve": [100, 102, 101, 105],
        "trade_markers": [{"step": 0, "price": 100, "action": "buy"},
                          {"step": 3, "price": 105, "action": "sell"}],
        "equity_curve": [1000, 1000, 1010, 1050],
        "benchmark_curve": [1000, 1020, 1010, 1050],
    }

    def test_two_panels_price_and_equity(self):
        fig = backtest_figure(self.RESULT)
        assert isinstance(fig, Figure) and len(fig.axes) == 2

    def test_price_panel_marks_buy_and_sell(self):
        ax = backtest_figure(self.RESULT).axes[0]
        labels = [c.get_label() for c in ax.collections]  # scatter layers
        assert "buy" in labels and "sell" in labels

    def test_handles_no_trades(self):
        empty = {**self.RESULT, "trade_markers": []}
        fig = backtest_figure(empty)  # no markers → still a valid 2-panel figure
        assert len(fig.axes) == 2


class TestComparisonFigure:
    def test_one_labelled_line_per_architecture(self):
        fig = comparison_figure(
            {"Dueling DQN": [{"reward": 1.0}, {"reward": 2.0}], "Plain DQN": [{"reward": 0.5}]}
        )
        labels = [line.get_label() for line in fig.axes[0].get_lines()]
        assert labels == ["Dueling DQN", "Plain DQN"]


class TestQValueFigure:
    def test_three_bars_with_chosen_highlighted(self):
        fig = q_value_figure([0.1, 0.2, 0.5], action_index=2, labels=["Sell", "Hold", "Buy"])
        ax = fig.axes[0]
        assert len(ax.patches) == 3            # Sell / Hold / Buy
        assert "Buy" in ax.get_title()         # chosen action named in the title
