"""Tests for ensure_tk_libraries (B22 — make the bundled Tcl/Tk discoverable).

The uv standalone Python ships Tcl/Tk under ``{base}/lib/tcl*`` + ``lib/tk*``
but cannot locate them; the helper points TCL_LIBRARY/TK_LIBRARY at them so
``tkinter.Tk()`` starts. These tests use a fake interpreter tree (no real Tk).
"""

from tradedqn.gui.tcl_setup import ensure_tk_libraries


def _fake_base(tmp_path, *, tcl="tcl9.0", tk="tk9.0"):
    lib = tmp_path / "lib"
    (lib / tcl).mkdir(parents=True)
    (lib / tcl / "init.tcl").write_text("# init")
    (lib / "tcl9").mkdir()  # decoy: a tcl* dir with no init.tcl must be skipped
    (lib / tk).mkdir()
    (lib / tk / "tk.tcl").write_text("# tk")
    return str(tmp_path)


class TestEnsureTkLibraries:
    def test_sets_both_vars_to_bundled_dirs(self, tmp_path):
        base = _fake_base(tmp_path)
        env: dict[str, str] = {}
        ensure_tk_libraries(base_prefix=base, environ=env)
        assert env["TCL_LIBRARY"].endswith("/lib/tcl9.0")
        assert env["TK_LIBRARY"].endswith("/lib/tk9.0")

    def test_does_not_override_existing_value(self, tmp_path):
        base = _fake_base(tmp_path)
        env = {"TCL_LIBRARY": "/custom/tcl"}
        ensure_tk_libraries(base_prefix=base, environ=env)
        assert env["TCL_LIBRARY"] == "/custom/tcl"  # user/system override respected
        assert env["TK_LIBRARY"].endswith("/lib/tk9.0")

    def test_no_op_when_no_bundled_dirs(self, tmp_path):
        (tmp_path / "lib").mkdir()
        env: dict[str, str] = {}
        ensure_tk_libraries(base_prefix=str(tmp_path), environ=env)
        assert env == {}  # nothing found, nothing set, no error
