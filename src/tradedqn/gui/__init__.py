"""GUI layer — a Tkinter + matplotlib dashboard over the TradingSDK.

Exports the testable pieces (controller + chart builders). The Tk window
(``app.MainWindow``) is imported lazily by ``main.py`` so importing this
package never requires a display.
"""

from tradedqn.gui.charts import equity_figure, training_figure
from tradedqn.gui.controller import GuiController

__all__ = ["GuiController", "equity_figure", "training_figure"]
