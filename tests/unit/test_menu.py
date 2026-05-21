"""Tests for the TerminalApp menu (B21 — presentation-only, SDK-driven)."""

from tradedqn.cli.menu import TerminalApp
from tradedqn.config import Config


class FakeSDK:
    """Stub SDK that records calls and returns canned results."""

    def __init__(self, fail_train=False):
        self.cfg = Config({"paths": {"checkpoint": "results/checkpoints/dqn.pt"}})
        self.calls: list[str] = []
        self._fail_train = fail_train

    def prepare_data(self):
        self.calls.append("prepare")
        return {"train": 100, "validation": 20, "test": 20}

    def train(self, episodes=None):
        self.calls.append("train")
        if self._fail_train:
            raise RuntimeError("call prepare_data() first")
        return [{"episode": 0, "epsilon": 0.5, "final_value": 1010.0}]

    def backtest(self, split="test"):
        self.calls.append("backtest")
        return {"total_return": 0.12, "benchmark_return": 0.08, "sharpe_ratio": 1.3,
                "max_drawdown": 0.05, "win_rate": 0.6, "num_trades": 4}

    def recommend(self, split="test"):
        self.calls.append("recommend")
        return {"action": "buy", "action_index": 2, "q_values": [0.1, 0.2, 0.3]}

    def save_brain(self, path):
        self.calls.append("save")

    def load_brain(self, path):
        self.calls.append("load")


def run_with(sdk, choices):
    out: list[str] = []
    responses = iter(choices)
    TerminalApp(sdk, input_fn=lambda _="": next(responses), output_fn=out.append).run()
    return "\n".join(out)


class TestDispatch:
    def test_full_pipeline_in_order(self):
        sdk = FakeSDK()
        run_with(sdk, ["1", "2", "3", "4", "5", "6", "0"])
        assert sdk.calls == ["prepare", "train", "backtest", "recommend", "save", "load"]

    def test_quit_is_clean(self):
        out = run_with(FakeSDK(), ["0"])
        assert "Bye." in out

    def test_menu_lists_all_options(self):
        out = run_with(FakeSDK(), ["0"])
        for label in ("Prepare data", "Train", "Backtest", "Recommend", "Quit"):
            assert label in out

    def test_unknown_option_loops(self):
        sdk = FakeSDK()
        out = run_with(sdk, ["9", "0"])
        assert "Unknown option" in out and sdk.calls == []


class TestOutputAndErrors:
    def test_backtest_output_has_numbers(self):
        out = run_with(FakeSDK(), ["3", "0"])
        assert "total_return=12.00%" in out and "trades=4" in out

    def test_recommend_output_has_action(self):
        out = run_with(FakeSDK(), ["4", "0"])
        assert "BUY" in out

    def test_sdk_error_is_caught_not_raised(self):
        out = run_with(FakeSDK(fail_train=True), ["2", "0"])
        assert "Error: call prepare_data() first" in out
