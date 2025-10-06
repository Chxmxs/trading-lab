# companion/explorer/dashboard.py
from __future__ import annotations

# We try Dash 2.x first, then Dash 1.x component packages.
# If neither is available (CI), we provide tiny stubs so imports/tests succeed.
HAVE_DASH = False
Dash = None  # type: ignore
try:
    from dash import Dash, html, dcc  # Dash 2.x
    HAVE_DASH = True
except Exception:
    try:
        import dash  # noqa: F401
        import dash_core_components as dcc  # type: ignore
        import dash_html_components as html  # type: ignore
        HAVE_DASH = True
    except Exception:
        # Minimal stubs to satisfy tests without the dash package installed.
        class _HTML:
            @staticmethod
            def Div(children=None, **props):
                return {"component": "Div", "props": props, "children": children or []}
            @staticmethod
            def H1(children=None, **props):
                return {"component": "H1", "props": props, "children": children or []}
            @staticmethod
            def P(children=None, **props):
                return {"component": "P", "props": props, "children": children or []}
        class _DCC:
            @staticmethod
            def Graph(**props):
                return {"component": "Graph", "props": props}
            @staticmethod
            def Dropdown(**props):
                return {"component": "Dropdown", "props": props}
        html, dcc = _HTML(), _DCC()  # type: ignore

def build_layout(title: str = "Trading-Lab Explorer"):
    """
    Returns a minimal dashboard layout.
    Works with real Dash or with internal stubs when Dash is not installed.
    Tests import this function; avoid side effects at module import time.
    """
    return html.Div([
        html.H1(title, id="title"),
        html.Div([
            dcc.Dropdown(id="strategy-select", options=[], placeholder="Select strategy"),
            dcc.Dropdown(id="symbol-select", options=[], placeholder="Select symbol"),
        ], id="controls"),
        html.Div([
            dcc.Graph(id="equity-curve"),
            dcc.Graph(id="perf-matrix"),
        ], id="charts"),
        html.P("Dash available: {}".format("yes" if HAVE_DASH else "no"), id="dash-flag")
    ], id="root")

def create_app():
    """
    Only construct a Dash app if Dash is installed.
    Useful for local runs: `python -m companion.explorer.dashboard`.
    """
    if not HAVE_DASH:
        raise RuntimeError("Dash is not installed in this environment.")
    app = Dash(__name__)
    app.layout = build_layout()
    return app

if __name__ == "__main__":
    if not HAVE_DASH:
        raise SystemExit("Dash not installed. pip install dash && rerun.")
    create_app().run_server(debug=True)
