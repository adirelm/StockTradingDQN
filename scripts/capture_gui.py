"""Capture a real screenshot of the Tk GUI (the §10/§22 screenshot-regression aid).

Launches the actual ``MainWindow``, runs a short real pipeline so the dashboard
shows a genuine equity curve + metrics, then grabs the window to a PNG via the
macOS ``screencapture`` tool. Re-run anytime the GUI changes:

    uv run python scripts/capture_gui.py            # docs/assets/gui_dashboard.png
    uv run python scripts/capture_gui.py --episodes 20 --out /tmp/gui.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tradedqn.gui.app import MainWindow
from tradedqn.sdk import TradingSDK

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "docs" / "assets" / "gui_dashboard.png"


def _populate(window: MainWindow, episodes: int, view: str) -> None:
    """Drive the GUI to the requested view so its chart/state renders for capture."""
    window.status.set(window.controller.prepare())
    window._episodes.set(episodes)
    if view == "compare":
        window._do_compare()            # Dueling-vs-plain reward overlay
    elif view == "train":
        window._do_train()              # live-animating reward + ε chart
    else:                               # backtest / recommend need a trained agent first
        window.controller.sdk.train(episodes=episodes)
        (window._do_recommend if view == "recommend" else window._do_backtest)()


def _grab(window: MainWindow, out_path: Path) -> None:
    root = window.root
    region = f"{root.winfo_rootx()},{root.winfo_rooty()},{root.winfo_width()},{root.winfo_height()}"
    # Grabbing from the main thread (after mainloop returns) — not inside a Tk
    # callback, where screencapture intermittently fails on this machine.
    subprocess.run(["screencapture", "-x", "-R", region, str(out_path)], check=True)


def capture(out_path: Path, episodes: int, view: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window = MainWindow(TradingSDK())
    window.root.geometry("1024x720+60+60")
    _populate(window, episodes, view)
    window.root.deiconify()
    window.root.lift()
    window.root.attributes("-topmost", True)
    window.root.after(700, window.root.quit)  # let the WM composite, then leave mainloop
    window.root.mainloop()
    _grab(window, out_path)  # window still on screen here
    window.root.destroy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture the TradeDQN GUI to a PNG.")
    parser.add_argument("--episodes", type=int, default=3, help="short train length")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG path")
    parser.add_argument(
        "--view", default="backtest",
        choices=["backtest", "train", "recommend", "compare"], help="which GUI state to capture",
    )
    args = parser.parse_args(argv)
    capture(args.out, args.episodes, args.view)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
