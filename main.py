"""Entry point — launch the TradeDQN terminal interface over the SDK."""

from tradedqn.cli.menu import TerminalApp
from tradedqn.sdk import TradingSDK


def main() -> None:
    TerminalApp(TradingSDK()).run()


if __name__ == "__main__":
    main()
