"""
tests/conftest.py

Stubs out Gradio and the Gradio app so that importing `server` during
pytest collection does not trigger network calls (GoogleFont download, etc.)
and does not hang.

This conftest is loaded automatically by pytest before any test module,
so by the time test_api.py does `from server import app` the stubs are
already in sys.modules.
"""
from __future__ import annotations

import sys
import types


# ---------------------------------------------------------------------------
# 1. Build a minimal gradio stub that satisfies server.py + app.py at import
#    time without making any network requests or launching a server.
# ---------------------------------------------------------------------------

def _make_gradio_stub() -> types.ModuleType:
    gr = types.ModuleType("gradio")

    # --- Themes sub-module --------------------------------------------------
    themes_mod = types.ModuleType("gradio.themes")

    class _FakeThemeBase:
        def __init__(self, *a, **kw):
            pass

    class Soft(_FakeThemeBase):
        pass

    class GoogleFont:
        def __init__(self, *a, **kw):
            pass

    themes_mod.Soft = Soft
    themes_mod.GoogleFont = GoogleFont
    themes_mod.Base = _FakeThemeBase
    sys.modules["gradio.themes"] = themes_mod
    gr.themes = themes_mod  # type: ignore[attr-defined]

    # --- gr.Blocks context manager -----------------------------------------
    class _FakeBlocks:
        """A no-op context manager that acts like gr.Blocks(...)."""

        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def launch(self, **kw):
            pass

    gr.Blocks = _FakeBlocks  # type: ignore[attr-defined]

    # --- common widgets used in app.py -------------------------------------
    for name in (
        "Markdown", "Row", "Column", "Dropdown", "Button", "Radio",
        "Slider", "Accordion", "Textbox", "HTML",
    ):
        cls = type(name, (), {"__init__": lambda self, *a, **kw: None})
        setattr(gr, name, cls)

    # --- gr.mount_gradio_app -----------------------------------------------
    def _mount_gradio_app(fastapi_app, gradio_app, path="/"):
        """Return the FastAPI app unchanged — no actual mounting."""
        return fastapi_app

    gr.mount_gradio_app = _mount_gradio_app  # type: ignore[attr-defined]

    return gr


# Install the stub *before* any test file is imported.
_gr_stub = _make_gradio_stub()
sys.modules.setdefault("gradio", _gr_stub)


# ---------------------------------------------------------------------------
# 2. Install a stub for `app` (app.py) so `from app import demo` succeeds
#    without executing the real Gradio blocks.
# ---------------------------------------------------------------------------

_app_stub = types.ModuleType("app")
_app_stub.demo = _gr_stub.Blocks()  # a harmless _FakeBlocks instance
sys.modules.setdefault("app", _app_stub)

# ---------------------------------------------------------------------------
# 3. Also stub `server.app` so any import of that canonical path works too.
# ---------------------------------------------------------------------------

# We need a "server" package stub first (can't conflict with real server.py)
_server_pkg_stub = types.ModuleType("server")
_server_pkg_stub.app = None  # placeholder; populated below
sys.modules.setdefault("server", _server_pkg_stub)

_server_app_stub = types.ModuleType("server.app")
_server_app_stub.app = None  # the FastAPI app — filled in after server import
sys.modules.setdefault("server.app", _server_app_stub)

