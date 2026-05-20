import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    ACCENT_ORANGE,
    load_data,
    load_raw_count,
    apply_sidebar_filters,
    render_page_header,
    render_dq_grid,
    section,
    finalise_fig,
)

st.set_page_config(
    page_title="Overview",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

render_page_header("overview", df)

# Data quality summary
with st.expander("Data quality summary", icon=":material/fact_check:"):
    raw_count = load_raw_count()
    removed = raw_count - len(df)
    removed_pct = (1 - len(df) / raw_count) * 100
    guest_rows = int(df["is_guest"].sum())
    guest_pct = df["is_guest"].mean() * 100
    render_dq_grid([
        ("Raw rows", f"{raw_count:,}"),
        ("After cleaning", f"{len(df):,}"),
        ("Rows removed", f"{removed:,} ({removed_pct:.1f}%)"),
        ("Unique customers", f"{df['Customer ID'].nunique():,}"),
        ("Date range", f"{df['InvoiceDate'].min():%d %b %Y} - {df['InvoiceDate'].max():%d %b %Y}"),
        ("Guest checkouts", f"{guest_rows:,} ({guest_pct:.1f}%)"),
        ("Registered customers", f"{(~df['is_guest']).sum():,}"),
    ])
    st.caption(
        f"Note: {guest_pct:.1f}% of transactions are guest checkouts (no Customer ID). "
        "Revenue figures include all transactions. RFM segmentation uses registered customers only."
    )
# KPI row — restricted to the trailing 12 months of the filtered range so the
# eyebrow, headline values and sparklines all describe the same window.
max_date = df["InvoiceDate"].max()
t12m_start = max_date - pd.DateOffset(months=12)
df_t12 = df[df["InvoiceDate"] >= t12m_start]
invoice_totals_t12 = df_t12.groupby("Invoice")["Revenue"].sum()

section("Headline KPIs", eyebrow="Trailing 12 months")
monthly_kpi = df_t12.groupby("Month").agg(
    revenue=("Revenue", "sum"),
    customers=("Customer ID", "nunique"),
    orders=("Invoice", "nunique"),
).reset_index()
# AOV: mean of invoice totals within each month — matches the headline definition.
monthly_aov = (
    df_t12.groupby(["Month", "Invoice"])["Revenue"].sum()
    .groupby(level="Month").mean()
    .rename("aov").reset_index()
)
monthly_kpi = monthly_kpi.merge(monthly_aov, on="Month", how="left")

trend_revenue = monthly_kpi["revenue"].tail(12).tolist()
trend_customers = monthly_kpi["customers"].tail(12).tolist()
trend_orders = monthly_kpi["orders"].tail(12).tolist()
trend_aov = monthly_kpi["aov"].tail(12).fillna(0).tolist()

guest_share_t12 = df_t12["is_guest"].mean() * 100

with st.container(horizontal=True):
    st.metric(
        "Revenue (T12M)",
        f"£{df_t12['Revenue'].sum():,.0f}",
        border=True,
        chart_data=trend_revenue,
        chart_type="line",
        help=f"Sum of revenue from {t12m_start:%d %b %Y} to {max_date:%d %b %Y}. Includes guest checkouts.",
    )
    st.metric(
        "Registered customers (T12M)",
        f"{df_t12['Customer ID'].nunique():,}",
        border=True,
        chart_data=trend_customers,
        chart_type="line",
        help=(
            "Distinct Customer IDs in the trailing 12 months. Guest checkouts "
            f"have no ID and so are excluded ({guest_share_t12:.1f}% of rows are guest)."
        ),
    )
    st.metric(
        "Orders (T12M)",
        f"{df_t12['Invoice'].nunique():,}",
        border=True,
        chart_data=trend_orders,
        chart_type="line",
        help="Distinct invoices in the trailing 12 months. Includes guest checkouts.",
    )
    st.metric(
        "Avg order value (T12M)",
        f"£{invoice_totals_t12.mean():,.2f}",
        border=True,
        chart_data=trend_aov,
        chart_type="line",
        help="Mean of invoice totals over the trailing 12 months. Sparkline shows the same mean computed within each month.",
    )

st.space("small")

# Monthly revenue area chart with smoothing
section("Revenue trend", eyebrow="Monthly")
monthly_revenue = df.groupby("Month")["Revenue"].sum().reset_index()
fig_line = go.Figure(go.Scatter(
    x=monthly_revenue["Month"],
    y=monthly_revenue["Revenue"],
    mode="lines",
    line=dict(color="#B85F3D", width=2.5, shape="spline"),
    fill="tozeroy",
    fillcolor="rgba(184,95,61,0.12)",
))
fig_line.update_layout(
    yaxis_title="Revenue (£)",
    yaxis_tickprefix="£",
    yaxis_tickformat=",",
    showlegend=False,
)
finalise_fig(fig_line, unified_hover=True)
st.plotly_chart(fig_line, width='stretch')

st.space("small")
section("Top performers", eyebrow="By revenue")
col_left, col_right = st.columns(2)
with col_left:
    top_countries = (
        df.groupby("Country")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig_bar = px.bar(
        top_countries, x="Revenue", y="Country",
        orientation="h", title="Top 10 countries by revenue",
        color_discrete_sequence=[ACCENT_ORANGE],
    )
    fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"), dragmode=False)
    fig_bar.update_xaxes(tickprefix="£", tickformat=",")
    finalise_fig(fig_bar)
    st.plotly_chart(fig_bar, width='stretch')

with col_right:
    # Group by StockCode (stable product identity) and use the most common
    # description as the display label — Description has typos/casing variants
    # that would otherwise split or merge SKUs incorrectly.
    top_products = (
        df.groupby("StockCode")
        .agg(
            Revenue=("Revenue", "sum"),
            Description=(
                "Description",
                lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "",
            ),
        )
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )
    fig_prod = px.bar(
        top_products, x="Revenue", y="Description",
        orientation="h", title="Top 10 products by revenue",
        color_discrete_sequence=[ACCENT_ORANGE],
    )
    fig_prod.update_layout(yaxis=dict(categoryorder="total ascending"), dragmode=False)
    fig_prod.update_xaxes(tickprefix="£", tickformat=",")
    finalise_fig(fig_prod)
    st.plotly_chart(fig_prod, width='stretch')

st.space("small")
with st.expander("View raw data sample", icon=":material/table_view:"):
    st.dataframe(
        df.head(500),
        width="stretch",
        column_config={
            "Revenue": st.column_config.NumberColumn(format="£ %.2f"),
            "Price": st.column_config.NumberColumn(format="£ %.2f"),
            "Quantity": st.column_config.NumberColumn(format="%,d"),
            "InvoiceDate": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
