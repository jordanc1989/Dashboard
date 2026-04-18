import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    ACCENT_ORANGE,
    load_data,
    load_raw_count,
    apply_sidebar_filters,
    render_dataset_subtitle,
    finalize_fig,
)

st.set_page_config(
    page_title="Overview · Customer analytics",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

render_dataset_subtitle(df)

# ── Data quality summary ──────────────────────────────────────────────
with st.expander("Data quality summary"):
    raw_count = load_raw_count()
    st.markdown(f"""
    | Check | Result |
    |---|---|
    | Raw rows | {raw_count:,} |
    | After cleaning | {len(df):,} |
    | Rows removed | {raw_count - len(df):,} ({(1 - len(df)/raw_count)*100:.1f}%) |
    | Unique customers | {df['Customer ID'].nunique():,} |
    | Date range | {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()} |
    | Guest checkout rows | {df['is_guest'].sum():,} ({df['is_guest'].mean()*100:.1f}%) |
    | Registered customer rows | {(~df['is_guest']).sum():,} |
    """)
    guest_pct = df["is_guest"].mean() * 100
    st.caption(
        f"Note: {guest_pct:.1f}% of transactions are guest checkouts (no Customer ID). "
        "Revenue figures include all transactions. RFM segmentation uses registered customers only."
    )
# ── KPI row ───────────────────────────────────────────────────────────
monthly = df.groupby("Month").agg(
    revenue=("Revenue", "sum"),
    customers=("Customer ID", "nunique"),
    orders=("Invoice", "nunique"),
).reset_index()
monthly["aov"] = monthly["revenue"] / monthly["orders"].replace(0, pd.NA)

trend_revenue = monthly["revenue"].tail(12).tolist()
trend_customers = monthly["customers"].tail(12).tolist()
trend_orders = monthly["orders"].tail(12).tolist()
trend_aov = monthly["aov"].tail(12).fillna(0).tolist()

with st.container(horizontal=True):
    st.metric(
        "Total revenue",
        f"£{df['Revenue'].sum():,.0f}",
        border=True,
        chart_data=trend_revenue,
        chart_type="line",
    )
    st.metric(
        "Unique customers",
        f"{df['Customer ID'].nunique():,}",
        border=True,
        chart_data=trend_customers,
        chart_type="line",
    )
    st.metric(
        "Total orders",
        f"{df['Invoice'].nunique():,}",
        border=True,
        chart_data=trend_orders,
        chart_type="line",
    )
    st.metric(
        "Avg order value",
        f"£{df.groupby('Invoice')['Revenue'].sum().mean():,.2f}",
        border=True,
        chart_data=trend_aov,
        chart_type="line",
    )

# ── Monthly revenue — area chart with spline smoothing ────────────────
monthly_revenue = df.groupby("Month")["Revenue"].sum().reset_index()
fig_line = go.Figure(go.Scatter(
    x=monthly_revenue["Month"],
    y=monthly_revenue["Revenue"],
    mode="lines",
    line=dict(color="#2C78B7", width=2.5, shape="spline"),
    fill="tozeroy",
    fillcolor="rgba(44,120,183,0.12)",
))
fig_line.update_layout(
    title="Monthly revenue trend",
    yaxis_title="Revenue (£)",
    yaxis_tickprefix="£",
    yaxis_tickformat=",",
    showlegend=False,
)
finalize_fig(fig_line, unified_hover=True)
st.plotly_chart(fig_line, width='stretch')

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
    fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig_bar.update_xaxes(tickprefix="£", tickformat=",")
    finalize_fig(fig_bar)
    st.plotly_chart(fig_bar, width='stretch')

with col_right:
    top_products = (
        df.groupby("Description")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig_prod = px.bar(
        top_products, x="Revenue", y="Description",
        orientation="h", title="Top 10 products by revenue",
        color_discrete_sequence=[ACCENT_ORANGE],
    )
    fig_prod.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig_prod.update_xaxes(tickprefix="£", tickformat=",")
    finalize_fig(fig_prod)
    st.plotly_chart(fig_prod, width='stretch')

with st.expander("View raw data sample"):
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
