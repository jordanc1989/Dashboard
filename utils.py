import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import boxcox

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


SEGMENT_COLORS = {
    "A":           "#B85F3D",
    "B":     "#2E7D68",
    "C": "#7A52B3",
    "D":           "#2C78B7",
    "E":      "#B6861E",
    "F":             "#433D37"
}

SEGMENT_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F"
]


def segment_labels_for_k(k: int) -> list[str]:
    """Return k evenly-spaced labels from SEGMENT_LABELS, best → worst."""
    n = len(SEGMENT_LABELS)
    indices = np.round(np.linspace(0, n - 1, k)).astype(int)
    return [SEGMENT_LABELS[i] for i in indices]


def render_dataset_subtitle(df: pd.DataFrame) -> None:
    """Standard date-range line under page titles (matches Streamlit body tone)."""
    st.caption(
        f"UCI Online Retail Dataset: "
        f"{df['InvoiceDate'].min():%b %Y} - {df['InvoiceDate'].max():%b %Y}"
    )


# ── Page chrome & shared layout helpers ─────────────────────────────────────
# Palette + typography already live in .streamlit/config.toml. These helpers
# add visual hierarchy on top.

_PAGE_CHROME_CSS = """
<style>
/* Prevent text highlighting when expanding/collapsing Streamlit expanders */
summary {
    user-select: none;
}

/* Tabular numerals everywhere numbers need to align (metrics, tables, captions).
   Makes KPI rows and column ledgers read like a report, not a spreadsheet. */
[data-testid="stMetric"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stDataFrame"],
.page-header-meta,
.dq-grid dd {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1, "lnum" 1;
}

/* Page title: consistent letter-spacing and tight line-height across all pages.
   Streamlit renders st.title() as a sibling component, not a child of the
   injected page-header-block div, so we target the Streamlit heading wrapper. */
[data-testid="stHeadingWithActionElements"] h1 {
    letter-spacing: -0.018em !important;
    line-height: 1.08 !important;
    margin-bottom: 0.2rem !important;
}

/* Page header lede, rule, and meta are st.html() siblings (not children of the
   header wrapper div), so selectors target the elements directly. */
.page-header-lede {
    color: #5c5642;
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 72ch;
    margin: 0.25rem 0 0.35rem 0;
}
.page-header-meta {
    color: #6a6350;
    font-size: 0.82rem;
    letter-spacing: 0.01em;
    margin: 0;
}
.page-hero-tagline {
    color: #5c5642;
    font-size: 1.08rem;
    line-height: 1.55;
    max-width: 62ch;
    margin: 0.25rem 0 0 0;
}

/* Section header: small uppercase eyebrow + bold title. */
div.section-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin: 1.5rem 0 0.65rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e8e2d2;
}
div.section-header .section-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.7rem;
    color: #8a7f66;
    font-weight: 600;
}
div.section-header .section-title {
    font-size: 1.12rem;
    font-weight: 600;
    color: #2b2718;
    letter-spacing: -0.005em;
}

/* Metric cards: soften the hard border, tighten the numeral rhythm, muted label. */
[data-testid="stMetric"] {
    border-color: #e8e2d2 !important;
    background: rgba(253,253,248,0.6);
    padding: 0.9rem 1rem 0.85rem 1rem !important;
    box-shadow: none !important;
    transition: border-color 180ms ease, background-color 180ms ease;
}
[data-testid="stMetric"]:hover {
    border-color: #d8cfb8 !important;
    background: rgba(253,253,248,0.9);
}
[data-testid="stMetricLabel"] p {
    color: #8a7f66 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #2b2718 !important;
    font-weight: 500 !important;
    letter-spacing: -0.015em;
    line-height: 1.15 !important;
}

/* Expander chrome: softer surface, suppress default shadow.
   Border and border-radius are owned by the theme (baseRadius 0.75rem) on
   the inner <details> element — adding our own border here creates a double
   border with mismatched corner radii, so we leave those to the theme. */
[data-testid="stExpander"] {
    background: rgba(253,253,248,0.5);
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.92rem;
    color: #3d3a2a;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #B85F3D;
}

/* Data-quality definition list — used for summary blocks instead of a markdown table. */
dl.dq-grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    column-gap: 1.5rem;
    row-gap: 0.45rem;
    margin: 0.5rem 0 0.25rem 0;
    padding: 0;
}
dl.dq-grid dt {
    color: #8a7f66;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0;
    align-self: baseline;
    padding-top: 0.1rem;
}
dl.dq-grid dd {
    color: #2b2718;
    font-size: 0.95rem;
    margin: 0;
    border-bottom: 1px dotted #e8e2d2;
    padding-bottom: 0.4rem;
}
dl.dq-grid dd:last-of-type { border-bottom: none; }

/* Sidebar "Filters" eyebrow spacing. */
section[data-testid="stSidebar"] .sidebar-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.7rem;
    color: #8a7f66;
    font-weight: 600;
    margin: 0.25rem 0 0.45rem 0;
}

/* Captions: slightly muted, generous line-height for readability. */
[data-testid="stCaptionContainer"] p {
    color: #7a7060 !important;
    line-height: 1.55;
}

/* Bordered containers: very subtle warm tint distinguishes interactive control
   panels and cards from the plain page background. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(236,228,210,0.22) !important;
    transition: border-color 200ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #d0c9b5 !important;
}

/* Streamlit download / primary button tone, keep cursor consistent. */
button[kind="secondary"], button[kind="primary"] {
    cursor: pointer;
    transition: background-color 180ms ease, border-color 180ms ease;
}
</style>
"""


def inject_page_chrome() -> None:
    """Inject shared page chrome styles once per session.

    Idempotent across reruns (same markup), so calling it at the top of
    every page is safe.
    """
    st.html(_PAGE_CHROME_CSS)


# Canonical page metadata — keeps eyebrow/icon/lede in one place so home
# cards and page headers stay in sync.
PAGE_META: dict[str, dict[str, str]] = {
    "overview": {
        "eyebrow": "Exploration",
        "eyebrow_color": "blue",
        "icon": ":material/analytics:",
        "title": "Overview",
        "lede": (
            "Headline revenue, customer and order KPIs across the UCI Online Retail "
            "II dataset, with monthly trend and top-line breakdowns by country "
            "and product."
        ),
    },
    "rfm": {
        "eyebrow": "Segmentation",
        "eyebrow_color": "violet",
        "icon": ":material/hub:",
        "title": "RFM segmentation",
        "lede": (
            "Customers are scored on Recency (days since last purchase), Frequency "
            "(orders), and Monetary (total spend). K-means groups them intelligently into "
            "segments for targeted marketing."
        ),
    },
    "churn": {
        "eyebrow": "Classification",
        "eyebrow_color": "orange",
        "icon": ":material/trending_down:",
        "title": "Churn prediction",
        "lede": (
            "Non-contractual retail has no explicit churn signal, so we fabricate "
            "one from a time split. A random forest model then predicts which registered "
            "customers are drifting towards dormancy."
        ),
    },
    "clv": {
        "eyebrow": "Lifetime value",
        "eyebrow_color": "green",
        "icon": ":material/payments:",
        "title": "CLV prediction",
        "lede": (
            "Two probabilistic models are chained: BG/NBD forecasts when customers "
            "will purchase again, and Gamma-Gamma forecasts how much they'll spend. "
            "Together they deliver a per-customer lifetime value."
        ),
    },
    "forecast": {
        "eyebrow": "Forecasting",
        "eyebrow_color": "gray",
        "icon": ":material/show_chart:",
        "title": "Revenue forecasting",
        "lede": (
            "Classical time-series models project future revenue from the historical "
            "transaction stream, with SARIMA and Theta models available."
        ),
    },
}


def render_page_header(
    page_key: str,
    df: pd.DataFrame | None = None,
    *,
    lede: str | None = None,
) -> None:
    """Consistent eyebrow → title → lede → meta header used on every page.

    Pass ``lede`` to override the canonical page lede from ``PAGE_META``.
    """
    inject_page_chrome()
    meta = PAGE_META[page_key]

    st.html('<div class="page-header-block">')
    top_cols = st.columns([3, 2])
    with top_cols[0]:
        st.badge(meta["eyebrow"], color=meta["eyebrow_color"], icon=meta["icon"])
    with top_cols[1]:
        if df is not None and len(df):
            st.html(
                '<p class="page-header-meta" style="text-align:right;">'
                f'UCI Online Retail II · '
                f'{df["InvoiceDate"].min():%b %Y} - {df["InvoiceDate"].max():%b %Y}'
                '</p>'
            )

    st.title(meta["title"], anchor=False)
    effective_lede = lede if lede is not None else meta.get("lede")
    if effective_lede:
        st.html(f'<p class="page-header-lede">{effective_lede}</p>')
    st.html('</div>')


def render_dq_grid(items: list[tuple[str, str]]) -> None:
    """Render a two-column definition list for data-quality / summary blocks.

    Replaces markdown tables where hairline dividers and uppercase labels read
    more like a report than a spreadsheet.
    """
    inject_page_chrome()
    rows = "".join(
        f"<dt>{label}</dt><dd>{value}</dd>" for label, value in items
    )
    st.html(f'<dl class="dq-grid">{rows}</dl>')


def section(title: str, eyebrow: str | None = None) -> None:
    """Editorial-style section header: small uppercase eyebrow + bold title.

    Replaces ``st.subheader`` where a stronger two-line rhythm is useful.
    """
    inject_page_chrome()
    eyebrow_html = (
        f'<span class="section-eyebrow" style="text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.7rem; color: #8a7f66; font-weight: 600; margin-right: 0.6rem;">{eyebrow}</span>' if eyebrow else ""
    )
    st.html(
        f'<div class="section-header" style="display: flex; align-items: baseline; margin: 1.5rem 0 0.65rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid #e8e2d2;">'
        f'{eyebrow_html}'
        f'<span class="section-title" style="font-size: 1.12rem; font-weight: 600; color: #2b2718; letter-spacing: -0.005em;">{title}</span>'
        f'</div>'
    )


# Shared retail CSV hygiene: non-product / fee lines and dubious geography labels.
INVALID_RETAIL_COUNTRIES = [
    "European Community",
    "Korea",
    "West Indies",
    "Unspecified",
]
DESCRIPTION_NOISE_TERMS = [
    "POSTAGE",
    "DOTCOM",
    "BANK CHARGES",
    "MANUAL",
    "AMAZONFEE",
    "CRUK",
    "SAMPLES",
    "TEST",
]


def _retail_csv_line_filters(df):
    """Drop fee/adjustment-style lines and invalid rows. Expects Invoice/StockCode as str."""
    out = df.copy()
    if pd.api.types.is_categorical_dtype(out["Country"]):
        if "Ireland" not in out["Country"].cat.categories:
            out["Country"] = out["Country"].cat.add_categories(["Ireland"])
    out["Country"] = out["Country"].replace({"EIRE": "Ireland"})
    out = out[~out["Country"].isin(INVALID_RETAIL_COUNTRIES)]
    out = out.dropna(subset=["Description"])
    out = out[out["Description"] != "Adjust bad debt"]
    # Remove rows that contain 'Adjustment' in the Description column
    out = out[~out["Description"].str.contains("Adjustment", na=False)]
    pattern = "|".join(DESCRIPTION_NOISE_TERMS)
    out = out[~out["Description"].str.upper().str.contains(pattern, na=False)]
    out = out[~out["StockCode"].str.upper().str.startswith("POST")]
    return out


def assign_segment_labels(rfm):
    stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    rng = (stats.max() - stats.min()).replace(0, 1)
    r_norm = (stats["Recency"] - stats["Recency"].min()) / rng["Recency"]
    f_norm = (stats["Frequency"] - stats["Frequency"].min()) / rng["Frequency"]
    m_norm = (stats["Monetary"] - stats["Monetary"].min()) / rng["Monetary"]
    score = (1 - r_norm) + f_norm + m_norm
    ranked = score.sort_values(ascending=False).index
    labels = segment_labels_for_k(len(ranked))
    label_map = {cid: labels[rank] for rank, cid in enumerate(ranked)}
    return rfm["Cluster"].map(label_map)


@st.cache_data(max_entries=1)
def _load_raw():
    """Single source of truth for the raw CSV read.

    Everything downstream (orders, cancels, raw counts) derives from this
    cached frame to avoid re-reading the file three times.
    """
    df = pd.read_csv(
        "data/online_retail_II.csv",
        usecols=[
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ],
        dtype={
            "Invoice": "string",
            "StockCode": "string",
            "Description": "string",
            "Quantity": "int32",
            "Price": "float32",
            "Customer ID": "float64",
            "Country": "category",
        },
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


@st.cache_data(max_entries=2)
def load_data():
    df = _retail_csv_line_filters(_load_raw())

    cancel_mask = df["Invoice"].str.startswith("C")
    cancels = df[cancel_mask].copy()
    orders = df[~cancel_mask].copy()

    if not cancels.empty:
        cancels["_qty_abs"] = cancels["Quantity"].abs()
        cancel_counts = (
            cancels.groupby(["Customer ID", "StockCode", "_qty_abs"])
            .size()
            .reset_index(name="_n_cancel")
            .rename(columns={"_qty_abs": "Quantity"})
        )
        orders["_cumcount"] = orders.groupby(
            ["Customer ID", "StockCode", "Quantity"], dropna=False
        ).cumcount()
        orders = orders.merge(
            cancel_counts, on=["Customer ID", "StockCode", "Quantity"], how="left"
        )
        orders["_n_cancel"] = orders["_n_cancel"].fillna(0).astype(int)
        orders = orders[orders["_cumcount"] >= orders["_n_cancel"]].drop(
            columns=["_cumcount", "_n_cancel"]
        )

    df = orders

    df["is_guest"] = df["Customer ID"].isna()
    df["Customer ID"] = df["Customer ID"].astype("Int64").astype("string")

    df = df[df["Quantity"] > 0]
    df = df[df["Price"] >= 0.01]
    df = df.drop_duplicates()

    df["Revenue"] = df["Quantity"] * df["Price"]
    df = df.dropna(subset=["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    return df


@st.cache_data(max_entries=1)
def load_raw_count():
    return len(_load_raw())


@st.cache_data(max_entries=2)
def load_cancels():
    """Load only cancel (return) invoices, which `load_data` strips out.

    Uses `_retail_csv_line_filters` like `load_data` so cancel counts stay
    comparable. Returned columns: Customer ID, InvoiceDate, Invoice.
    """
    df = _retail_csv_line_filters(_load_raw())

    df = df[df["Invoice"].str.startswith("C")]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype("Int64").astype("string")
    df = df.dropna(subset=["InvoiceDate"])
    return df[["Customer ID", "InvoiceDate", "Invoice"]]


@st.cache_data
def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    return rfm


@st.cache_data
def transform_rfm(rfm):
    rfm = rfm.copy()
    rfm["R_t"], _ = boxcox(rfm["Recency"] + 1)
    rfm["F_t"], _ = boxcox(rfm["Frequency"])
    rfm["M_t"], _ = boxcox(rfm["Monetary"])
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["R_t", "F_t", "M_t"]])
    return rfm, X


@st.cache_data(max_entries=16)
def run_clustering(rfm_raw, n_clusters, winsorise_pct=99):
    rfm = rfm_raw.copy()
    if winsorise_pct < 100:
        q = winsorise_pct / 100
        for col in ["Recency", "Frequency", "Monetary"]:
            rfm[col] = rfm[col].clip(lower=rfm[col].quantile(1 - q), upper=rfm[col].quantile(q))
    rfm, X = transform_rfm(rfm)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=10)
    rfm["Cluster"] = km.fit_predict(X)
    sil = silhouette_score(X, rfm["Cluster"]) if len(rfm) > n_clusters else float("nan")
    return rfm, sil


@st.cache_data(max_entries=16)
def elbow_data(rfm_raw, winsorise_pct=99, max_segments=6):
    rfm = rfm_raw.copy()

    if winsorise_pct < 100:
        q = winsorise_pct / 100
        for col in ["Recency", "Frequency", "Monetary"]:
            rfm[col] = rfm[col].clip(lower=rfm[col].quantile(1 - q), upper=rfm[col].quantile(q))

    _, X = transform_rfm(rfm)

    max_k = min(len(SEGMENT_LABELS), len(rfm) - 1)
    if max_k < 2:
        return [], [], []

    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    return list(k_range), inertias, silhouettes


@st.cache_data(max_entries=16)
def build_cohort(df):
    df = df.copy()
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")
    first_purchase = df.groupby("Customer ID")["OrderMonth"].min().reset_index()
    first_purchase.columns = ["Customer ID", "CohortMonth"]
    df = df.merge(first_purchase, on="Customer ID")
    df["PeriodNumber"] = (df["OrderMonth"] - df["CohortMonth"]).apply(lambda x: x.n)
    cohort_counts = (
        df.groupby(["CohortMonth", "PeriodNumber"])["Customer ID"]
        .nunique()
        .reset_index()
        .pivot(index="CohortMonth", columns="PeriodNumber", values="Customer ID")
    )
    cohort_sizes = cohort_counts[0]
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    retention.index = retention.index.astype(str)
    return retention, cohort_sizes.values


@st.cache_data(max_entries=16)
def build_revenue_series(df, freq="W"):
    """Aggregate transactions into a regular-interval revenue time series.

    freq:
      "W"  - week ending Sunday
      "MS" - month start
    Returns a pandas Series indexed by period with a DatetimeIndex at the
    requested frequency.
    """

    out = df.copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out = out.dropna(subset=["InvoiceDate", "Revenue"]).sort_values("InvoiceDate")

    series = (
        out.set_index("InvoiceDate")["Revenue"]
        .resample(freq)
        .sum()
        .rename("Revenue")
    )
    return series


@st.cache_data(max_entries=16)
def build_churn_dataset(df, churn_window_days=90):
    """Build a supervised churn dataset from transaction data.

    Non-contractual retail has no explicit churn label, so we fabricate one
    via a time split:

      cutoff = max_invoice_date - churn_window_days

    A registered customer who was active on or before `cutoff` is labelled
    **churned = 1** if they made no purchase in `(cutoff, max_date]` else 0.
    Features are computed strictly from transactions at or before the cutoff
    to avoid look-ahead leakage.

    Returns
    -------
    features : pd.DataFrame
        One row per customer, with engineered features and the `churned` label.
    meta : dict
        Context for display: cutoff date, window size, totals.
    """
    df = df[~df["is_guest"]].copy()
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=churn_window_days)

    pre = df[df["InvoiceDate"] <= cutoff]
    post = df[df["InvoiceDate"] > cutoff]

    invoices = (
        pre.groupby(["Customer ID", "Invoice", "InvoiceDate"])
        .agg(
            order_revenue=("Revenue", "sum"),
            order_items=("Quantity", "sum"),
            order_products=("StockCode", "nunique"),
        )
        .reset_index()
    )

    features = (
        invoices.groupby("Customer ID")
        .agg(
            recency_days=("InvoiceDate", lambda x: (cutoff - x.max()).days),
            tenure_days=("InvoiceDate", lambda x: (cutoff - x.min()).days),
            frequency=("Invoice", "nunique"),
            monetary=("order_revenue", "sum"),
            avg_order_value=("order_revenue", "mean"),
            avg_items_per_order=("order_items", "mean"),
            avg_unique_products=("order_products", "mean"),
        )
        .reset_index()
    )

    total_products = (
        pre.groupby("Customer ID")["StockCode"]
        .nunique()
        .rename("unique_products")
        .reset_index()
    )
    features = features.merge(total_products, on="Customer ID", how="left")

    tenure_months = (features["tenure_days"] / 30).clip(lower=1)
    features["orders_per_month"] = features["frequency"] / tenure_months

    cancels = load_cancels()
    cancel_counts = (
        cancels.loc[cancels["InvoiceDate"] <= cutoff]
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("n_returns")
        .reset_index()
    )
    features = features.merge(cancel_counts, on="Customer ID", how="left")
    features["n_returns"] = features["n_returns"].fillna(0).astype(int)
    features["return_rate"] = features["n_returns"] / (
        features["frequency"] + features["n_returns"]
    )

    active_post = set(post["Customer ID"].unique())
    features["churned"] = (~features["Customer ID"].isin(active_post)).astype(int)

    meta = {
        "cutoff": cutoff,
        "max_date": max_date,
        "window_days": churn_window_days,
        "n_customers": len(features),
        "churn_rate": features["churned"].mean(),
    }
    return features, meta


@st.cache_data(max_entries=8)
def build_clv_summary(df):
    """Build the per-customer RFM summary pymc-marketing expects.

    Columns returned (customer_id is a column, not the index):
      customer_id    = customer identifier
      frequency      = number of repeat transactions (total - 1)
      recency        = weeks from first to last purchase (model-internal definition;
                       display as 'weeks since last purchase' via T - recency)
      T              = weeks from first purchase to observation end
      monetary_value = mean revenue per repeat transaction
    """
    from pymc_marketing.clv.utils import rfm_summary

    obs_end = df["InvoiceDate"].max()

    invoices = (
        df.groupby(["Customer ID", "Invoice", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )

    summary = rfm_summary(
        invoices,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=obs_end,
        time_unit="W",
    )

    return summary.reset_index(drop=True)


def apply_sidebar_filters(df):
    with st.sidebar:
        st.html('<p class="sidebar-eyebrow">Filters</p>')
        with st.expander("Data scope", expanded=True, icon=":material/tune:"):
            countries = ["All"] + sorted(df["Country"].unique().tolist())
            selected_country = st.selectbox("Country", countries)
            if selected_country != "All":
                df = df[df["Country"] == selected_country]

            min_date = df["InvoiceDate"].min().date()
            max_date = df["InvoiceDate"].max().date()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter transactions by invoice date. RFM and cohort analyses will update accordingly.",
            )
    if len(date_range) == 2:
        df = df[
            (df["InvoiceDate"].dt.date >= date_range[0]) &
            (df["InvoiceDate"].dt.date <= date_range[1])
        ]
    else:
        st.sidebar.caption("Pick an end date to apply range.")
    return df
