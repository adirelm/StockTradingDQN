"""TerminalApp — a presentation-only menu over the TradingSDK.

Calls only the SDK (Prepare → Train → Backtest → Recommend, plus save/load).
``input_fn``/``output_fn`` are injected so it is testable and agent-drivable
(no real stdin). Handler errors are caught and surfaced — the menu never crashes.
"""

from __future__ import annotations

from collections.abc import Callable

from tradedqn.format import backtest_line, recommendation_line

MENU = (
    ("1", "Prepare data"),
    ("2", "Train"),
    ("3", "Backtest"),
    ("4", "Recommend next action"),
    ("5", "Save brain"),
    ("6", "Load brain"),
    ("0", "Quit"),
)


class TerminalApp:
    """Text menu over the SDK; injected I/O makes it testable and agent-drivable."""

    def __init__(self, sdk, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> None:
        self.sdk = sdk
        self._input = input_fn
        self._output = output_fn
        self._handlers = {
            "1": self._prepare,
            "2": self._train,
            "3": self._backtest,
            "4": self._recommend,
            "5": self._save,
            "6": self._load,
        }

    def run(self) -> None:
        """Loop: show the menu, dispatch the chosen handler, until the user quits."""
        while True:
            self._print_menu()
            choice = self._input("Select: ").strip()
            if choice == "0":
                self._output("Bye.")
                return
            handler = self._handlers.get(choice)
            if handler is None:
                self._output(f"Unknown option: {choice!r}")
                continue
            try:
                handler()
            except (RuntimeError, ValueError) as error:
                self._output(f"Error: {error}")

    def _print_menu(self) -> None:
        """Print the menu header and the numbered options."""
        self._output("\n=== TradeDQN ===")
        for key, label in MENU:
            self._output(f"  {key}. {label}")

    def _prepare(self) -> None:
        """Prepare data via the SDK and report the split sizes."""
        self._output(f"Prepared splits: {self.sdk.prepare_data()}")

    def _train(self) -> None:
        """Train via the SDK and report the last episode's stats."""
        last = self.sdk.train()[-1]
        self._output(
            f"Trained {last['episode'] + 1} episode(s); "
            f"ε={last['epsilon']:.3f}  final value={last['final_value']:.2f}"
        )

    def _backtest(self) -> None:
        """Run a backtest via the SDK and print the metrics line."""
        self._output(f"Backtest: {backtest_line(self.sdk.backtest())}")

    def _recommend(self) -> None:
        """Print the SDK's latest Buy/Hold/Sell recommendation."""
        self._output(recommendation_line(self.sdk.recommend()))

    def _save(self) -> None:
        """Save the agent checkpoint to the configured path."""
        path = self.sdk.cfg.paths.checkpoint
        self.sdk.save_brain(path)
        self._output(f"Saved brain → {path}")

    def _load(self) -> None:
        """Load the agent checkpoint from the configured path."""
        path = self.sdk.cfg.paths.checkpoint
        self.sdk.load_brain(path)
        self._output(f"Loaded brain ← {path}")
