"""Tests for the shared CLI/GUI presentation helpers (§3.2/§4.2 DRY, §6.1 matching file)."""

from tradedqn.format import backtest_line, recommendation_line


def test_recommendation_line_uppercases_action_and_lists_q():
    line = recommendation_line({"action": "buy", "q_values": [0.1, 0.2, 0.3]})
    assert "BUY" in line and "0.100" in line and "0.300" in line
    assert "confidence" not in line and "drivers" not in line  # optional fields absent


def test_recommendation_line_includes_confidence_and_drivers():
    line = recommendation_line(
        {"action": "buy", "q_values": [0.1, 0.2, 0.7],
         "confidence": 0.55, "top_features": ["rsi_14", "macd"]}
    )
    assert "confidence 55%" in line and "drivers: rsi_14, macd" in line


def test_backtest_line_formats_percentages_and_counts():
    line = backtest_line(
        {
            "total_return": 0.0448,
            "benchmark_return": 0.1208,
            "sharpe_ratio": 0.27,
            "max_drawdown": 0.178,
            "win_rate": 0.36,
            "num_trades": 72,
        }
    )
    assert "total_return=4.48%" in line
    assert "benchmark=12.08%" in line
    assert "Sharpe=0.27" in line and "trades=72" in line
