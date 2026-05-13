import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Aligned with [theme] chartCategoricalColors / Plotly portfolio colorway
CHART_COLORWAY = [
    "#B85F3D",
    "#2E7D68",
    "#7A52B3",
    "#2C78B7",
    "#B6861E",
    "#433D37",
]
ACCENT_ORANGE = "#fbae6f"
PRIMARY_ACCENT = "#ff8e32"

# Match `.streamlit/config.toml` [theme] font for Plotly chart chrome (code uses theme codeFont)
UI_FONT_FAMILY = "Outfit"

# Brand-aligned sequential scales for heatmaps
COLOR_SCALE_EXPECTED_PURCHASES = [
    "#f5f4ef",
    "#c5dde8",
    "#6ba8cf",
    "#2C78B7",
]
COLOR_SCALE_P_ALIVE = [
    "#8B5E52",
    "#ddd9ce",
    "#6a9e82",
    "#2E7D68",
]

pio.templates["portfolio"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=UI_FONT_FAMILY, size=12, color="#3d3a2a"),
        title=dict(
            font=dict(size=14, color="#141413", family=UI_FONT_FAMILY),
            x=0,
            xanchor="left",
            pad=dict(b=14),
        ),
        margin=dict(t=40, l=20, r=12, b=8),
        colorway=list(CHART_COLORWAY),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color="#5c5642"),
        ),
        hoverlabel=dict(
            bgcolor="#fdfdf8",
            bordercolor="#B85F3D",
            font=dict(family=UI_FONT_FAMILY, size=12, color="#2b2718"),
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#cec9bc",
            linewidth=1,
            ticks="outside",
            tickcolor="#cec9bc",
            ticklen=4,
            tickfont=dict(size=11, color="#6a6350"),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,138,120,0.14)",
            gridwidth=1,
            showline=False,
            zeroline=False,
            ticks="",
            tickfont=dict(size=11, color="#6a6350"),
        ),
    )
)
pio.templates.default = "plotly+portfolio"

# Warm neutrals for reference lines and grid chrome
NEUTRAL_GRID = "#b8ae98"
NEUTRAL_RADAR_GRID = "#dcd5c2"


def finalise_fig(fig, *, unified_hover: bool = False, uirevision: str | None = None):
    """Apply shared Plotly layout so all charts use the portfolio template consistently."""
    kwargs: dict = {"template": "plotly+portfolio"}
    if unified_hover:
        kwargs["hovermode"] = "x unified"
    if uirevision is not None:
        kwargs["uirevision"] = uirevision
    fig.update_layout(**kwargs)
    return fig
