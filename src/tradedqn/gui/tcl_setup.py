"""Make the interpreter's bundled Tcl/Tk discoverable before launching the GUI.

The uv-managed (python-build-standalone) interpreter ships Tcl/Tk under
``{base_prefix}/lib/tcl*`` and ``lib/tk*`` but bakes in the *build machine's*
search path, so ``tkinter.Tk()`` aborts with "Tcl wasn't installed properly".
Pointing ``TCL_LIBRARY``/``TK_LIBRARY`` at the bundled dirs fixes it. A system
or Homebrew Python that already resolves Tk needs no help, so we only fill in a
variable that is missing (an explicit value the user set is left untouched).
Called once, before the GUI creates its Tk root.
"""

from __future__ import annotations

import os
import sys
from glob import glob

# (env var, dir prefix under lib/, file that proves it is the real library dir)
_LIBRARIES = (("TCL_LIBRARY", "tcl", "init.tcl"), ("TK_LIBRARY", "tk", "tk.tcl"))


def _bundled_dir(base: str, prefix: str, marker: str) -> str | None:
    """Return the ``lib/{prefix}*`` dir that actually contains ``marker``."""
    for candidate in sorted(glob(os.path.join(base, "lib", f"{prefix}[0-9]*"))):
        if os.path.isfile(os.path.join(candidate, marker)):
            return candidate
    return None


def ensure_tk_libraries(base_prefix: str | None = None, environ=None) -> None:
    """Set TCL_LIBRARY/TK_LIBRARY to the bundled Tcl/Tk dirs when unset."""
    base = base_prefix if base_prefix is not None else sys.base_prefix
    env = environ if environ is not None else os.environ
    for var, prefix, marker in _LIBRARIES:
        if env.get(var):
            continue
        found = _bundled_dir(base, prefix, marker)
        if found:
            env[var] = found
