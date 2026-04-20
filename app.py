import pytensor
import pandas as pd
import streamlit as st

from utils import (
    PAGE_META,
    apply_sidebar_filters,
    inject_page_chrome,
    load_data,
    render_page_footer,
    section,
)

pytensor.config.cxx = ""  # Disables C compilation entirely - to fix current MacOS bug until Pytensor is updated

st.set_page_config(
    page_title="Customer analytics dashboard",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

st.logo("static/jordan_cheney_logo_dark.png", size="large")

PAGE_CARDS = [
    ("overview", "app_pages/1_Overview.py"),
    ("rfm", "app_pages/2_RFM_Segmentation.py"),
    ("churn", "app_pages/3_Churn_Prediction.py"),
    ("clv", "app_pages/4_CLV_Prediction.py"),
    ("forecast", "app_pages/5_Revenue_Forecasting.py"),
]


def _render_hero(df: pd.DataFrame) -> None:
    """Editorial hero for the home page: eyebrow, title, tagline, dataset meta."""
    inject_page_chrome()
    st.html('<div class="page-header-block page-hero-block">')

    st.title("Customer analytics dashboard", anchor=False)
    st.html(
        '<p class="page-hero-tagline">'
        'Interactive modules for exploring a two-year UK-based online retail '
        'dataset, from headline revenue KPIs to probabilistic CLV and time-series revenue forecasts.'
        '</p>'
    )
    st.html('</div>')


def _render_glance(df: pd.DataFrame) -> None:
    """Four at-a-glance KPIs"""
    total_revenue = float(df["Revenue"].sum())
    n_customers = int(df["Customer ID"].nunique())
    n_orders = int(df["Invoice"].nunique())
    span_days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days

    with st.container(horizontal=True):
        st.metric("Total revenue", f"£{total_revenue:,.0f}", border=True)
        st.metric("Unique customers", f"{n_customers:,}", border=True)
        st.metric("Invoices", f"{n_orders:,}", border=True)
        st.metric("Observation window", f"{span_days:,} days", border=True)


def _render_module_cards() -> None:
    """Module grid — each card: icon + eyebrow + title + description + open link."""
    for row_start in range(0, len(PAGE_CARDS), 2):
        cols = st.columns(2, gap="medium")
        for col, offset in zip(cols, (0, 1)):
            idx = row_start + offset
            if idx >= len(PAGE_CARDS):
                continue
            page_key, page_path = PAGE_CARDS[idx]
            meta = PAGE_META[page_key]
            with col:
                with st.container(border=True):
                    st.badge(
                        meta["eyebrow"],
                        color=meta["eyebrow_color"],
                        icon=meta["icon"],
                    )
                    st.markdown(f"#### {meta['title']}")
                    st.caption(meta["lede"])
                    st.page_link(
                        page_path,
                        label=f"Open {meta['title'].lower()}",
                        icon=":material/arrow_forward:",
                    )


def home_page():
    df = load_data()
    df = apply_sidebar_filters(df)

    _render_hero(df)
    st.space("medium")

    section("At a glance", eyebrow="Dataset snapshot")
    _render_glance(df)
    st.space("medium")

    section("Analyses", eyebrow="Jump to a module")
    st.caption(
        "Each module has its own controls and works off the same sidebar filters."
    )
    _render_module_cards()

    render_page_footer(df, note="Customer analytics · home")


navigation = st.navigation(
    [
        st.Page(home_page, title="Home", icon=":material/home:"),
        st.Page("app_pages/1_Overview.py", title="Overview", icon=":material/analytics:"),
        st.Page("app_pages/2_RFM_Segmentation.py", title="RFM segmentation", icon=":material/hub:"),
        st.Page("app_pages/3_Churn_Prediction.py", title="Churn prediction", icon=":material/trending_down:"),
        st.Page("app_pages/4_CLV_Prediction.py", title="CLV prediction", icon=":material/payments:"),
        st.Page("app_pages/5_Revenue_Forecasting.py", title="Revenue forecasting", icon=":material/show_chart:"),
    ],
    position="sidebar",
)

navigation.run()
