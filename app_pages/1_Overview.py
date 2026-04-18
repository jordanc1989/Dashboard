import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, load_raw_count, apply_sidebar_filters

st.set_page_config(
    page_title="Overview · Customer Analytics",
    page_icon="📈",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

st.title("Overview")
st.markdown(
    f"<p style='color:#6B7280;margin-top:-12px;font-size:0.95rem;'>"
    f"UCI Online Retail Dataset &nbsp;·&nbsp; "
    f"{df['InvoiceDate'].min().strftime('%b %Y')} - {df['InvoiceDate'].max().strftime('%b %Y')}"
    f"</p>",
    unsafe_allow_html=True,
)

# ── Data quality summary ──────────────────────────────────────────────
with st.expander("🧹 Data quality summary"):
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
    st.info(
        f"Note: {guest_pct:.1f}% of transactions are guest checkouts (no Customer ID). "
        "Revenue figures include all transactions. RFM segmentation uses registered customers only."
    )
# ── KPI row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{df['Revenue'].sum():,.0f}")
col2.metric("Unique Customers", f"{df['Customer ID'].nunique():,}")
col3.metric("Total Orders", f"{df['Invoice'].nunique():,}")
col4.metric("Avg Order Value", f"£{df.groupby('Invoice')['Revenue'].sum().mean():,.2f}")

st.divider()

# ── Monthly revenue — area chart with spline smoothing ────────────────
monthly_revenue = df.groupby("Month")["Revenue"].sum().reset_index()
fig_line = go.Figure(go.Scatter(
    x=monthly_revenue["Month"],
    y=monthly_revenue["Revenue"],
    mode="lines",
    line=dict(color="#2C78B7", width=2.5, shape="spline"),
    fill="tozeroy",
    fillcolor="rgba(29,78,216,0.08)",
))
fig_line.update_layout(
    title="Monthly Revenue Trend",
    yaxis_title="Revenue (£)",
    yaxis_tickprefix="£",
    yaxis_tickformat=",",
    showlegend=False,
)
st.plotly_chart(fig_line, width='stretch')

col_left, col_right = st.columns(2)
with col_left:
    top_countries = (
        df.groupby("Country")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig_bar = px.bar(
        top_countries, x="Revenue", y="Country",
        orientation="h", title="Top 10 Countries by Revenue",
        color_discrete_sequence=["#fbae6f"],
    )
    fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig_bar.update_xaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig_bar, width='stretch')

with col_right:
    top_products = (
        df.groupby("Description")["Revenue"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig_prod = px.bar(
        top_products, x="Revenue", y="Description",
        orientation="h", title="Top 10 Products by Revenue",
        color_discrete_sequence=["#fbae6f"],
    )
    fig_prod.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig_prod.update_xaxes(tickprefix="£", tickformat=",")
    st.plotly_chart(fig_prod, width='stretch')

with st.expander("🔍 View raw data sample"):
    st.dataframe(df.head(500), width='stretch')
