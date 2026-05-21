"""Entry point — TradeDQN terminal menu (default) or GUI dashboard (`gui`).

    uv run main.py        # terminal interface (lecturer: build/test this first)
    uv run main.py gui    # Tkinter + matplotlib dashboard
"""

import sys

from tradedqn.sdk import TradingSDK


def main() -> None:
    sdk = TradingSDK()
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        from tradedqn.gui.app import MainWindow

        MainWindow(sdk).run()
    else:
        from tradedqn.cli.menu import TerminalApp

        TerminalApp(sdk).run()


if __name__ == "__main__":
    main()
