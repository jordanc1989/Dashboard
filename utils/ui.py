import streamlit as st
import pandas as pd

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
   injected page-header-block div, so target the Streamlit heading wrapper. */
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

    Uses ``st.markdown`` with ``unsafe_allow_html=True`` because ``st.html``
    strips ``<style>`` tags in Streamlit 1.57+.
    """
    st.markdown(_PAGE_CHROME_CSS, unsafe_allow_html=True)


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
            "(orders), and Monetary (total spend). K-means is the baseline; the "
            "Compare tab benchmarks it against Gaussian mixture, Agglomerative "
            "clustering and HDBSCAN side by side."
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
            "Together, they deliver a per-customer lifetime value."
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


def render_dataset_subtitle(df: pd.DataFrame) -> None:
    """Standard date-range line under page titles (matches Streamlit body tone)."""
    st.caption(
        f"UCI Online Retail Dataset: "
        f"{df['InvoiceDate'].min():%b %Y} - {df['InvoiceDate'].max():%b %Y}"
    )


def render_page_header(
    page_key: str,
    df: pd.DataFrame | None = None,
    *,
    lede: str | None = None,
) -> None:
    """Consistent eyebrow -> title -> lede -> meta header used on every page.

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
    """Render a two-column definition list for data-quality / summary blocks."""
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
    eyebrow_html = f'<span class="section-eyebrow">{eyebrow}</span>' if eyebrow else ""
    st.html(
        f'<div class="section-header">{eyebrow_html}'
        f'<span class="section-title">{title}</span></div>'
    )


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
