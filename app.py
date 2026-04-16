import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import boxcox

# --- Page config ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Visual theme ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* KPI metric cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-top: 3px solid #1D4ED8;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
/* Tab bar */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
}
/* Dashboard headline */
h1 {
    font-family: "Futura", "Trebuchet MS", "Century Gothic", sans-serif !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly house style — applied to every figure via pio.templates.default ─────
_FONT = dict(family="system-ui, -apple-system, sans-serif", size=12, color="#374151")
pio.templates["portfolio"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(249,250,251,0.6)",
        font=_FONT,
        title=dict(font=dict(size=14, color="#111827"), x=0, xanchor="left", pad=dict(b=12)),
        margin=dict(t=52, l=12, r=12, b=12),
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False,
                   linecolor="#E5E7EB", tickfont=_FONT),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False,
                   linecolor="#E5E7EB", tickfont=_FONT),
        colorway=["#1D4ED8", "#059669", "#7C3AED", "#D97706",
                  "#DC2626", "#0891B2", "#65A30D", "#9333EA"],
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E5E7EB", borderwidth=1),
    )
)
pio.templates.default = "plotly+portfolio"

# ── Segment colour map (traffic-light: blue/green = healthy, amber/red = at risk)
SEGMENT_COLORS = {
    "Champions":           "#1D4ED8",
    "Loyal Customers":     "#059669",
    "Potential Loyalists": "#7C3AED",
    "Promising":           "#0891B2",
    "Need Attention":      "#D97706",
    "At Risk":             "#DC2626",
    "Hibernating":         "#6B7280",
    "Lost":                "#374151",
}

# --- Segment labels (ordered best to worst customer value) ---
SEGMENT_LABELS = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Promising",
    "Need Attention",
    "At Risk",
    "Hibernating",
    "Lost",
]

def assign_segment_labels(rfm):
    """Rank clusters by composite RFM score and assign business labels."""
    stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    rng = (stats.max() - stats.min()).replace(0, 1)
    r_norm = (stats["Recency"] - stats["Recency"].min()) / rng["Recency"]
    f_norm = (stats["Frequency"] - stats["Frequency"].min()) / rng["Frequency"]
    m_norm = (stats["Monetary"] - stats["Monetary"].min()) / rng["Monetary"]
    score = (1 - r_norm) + f_norm + m_norm  # higher = better
    ranked = score.sort_values(ascending=False).index
    label_map = {cid: SEGMENT_LABELS[rank] for rank, cid in enumerate(ranked)}
    return rfm["Cluster"].map(label_map)

# --- Data loading ---
@st.cache_data
def load_data():
    df = pd.read_excel("data/online_retail.xlsx", engine="openpyxl")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    # Change EIRE to Ireland for consistency with other country names
    df["Country"] = df["Country"].replace({"EIRE": "Ireland"})

    # Remove noise/internal transactions
    noise_terms = ["POSTAGE", "DOTCOM", "BANK CHARGES", "MANUAL",
                   "AMAZONFEE", "CRUK", "SAMPLES", "TEST"]
    pattern = "|".join(noise_terms)
    df = df[~df["Description"].str.upper().str.contains(pattern, na=False)]
    df = df.dropna(subset=["Description"])

    # Remove cancellations AND the original orders they cancelled.
    # Cancellation rows (InvoiceNo starts with "C") have negative Quantity.
    # For each (CustomerID, StockCode, abs_Quantity) we remove that many
    # matching original rows using cumcount() so we never over-remove.
    cancel_mask = df["InvoiceNo"].str.startswith("C")
    cancels = df[cancel_mask].copy()
    orders = df[~cancel_mask].copy()

    if not cancels.empty:
        cancels["_qty_abs"] = cancels["Quantity"].abs()
        cancel_counts = (
            cancels.groupby(["CustomerID", "StockCode", "_qty_abs"])
            .size()
            .reset_index(name="_n_cancel")
            .rename(columns={"_qty_abs": "Quantity"})
        )
        orders["_cumcount"] = orders.groupby(
            ["CustomerID", "StockCode", "Quantity"], dropna=False
        ).cumcount()
        orders = orders.merge(
            cancel_counts, on=["CustomerID", "StockCode", "Quantity"], how="left"
        )
        orders["_n_cancel"] = orders["_n_cancel"].fillna(0).astype(int)
        orders = orders[orders["_cumcount"] >= orders["_n_cancel"]].drop(
            columns=["_cumcount", "_n_cancel"]
        )

    df = orders

    # Flag guests before converting CustomerID so is_guest is meaningful.
    # Use Int64 (nullable integer) as intermediate so NaN survives the float→int
    # cast, then convert to string. Guests end up as pd.NA, not "nan".
    df["is_guest"] = df["CustomerID"].isna()
    df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")

    # Remove bad quantities / prices
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] >= 0.01]

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Derived columns
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    return df

@st.cache_data
def load_raw_count():
    raw = pd.read_excel("data/online_retail.xlsx", engine="openpyxl")
    return len(raw)

# --- RFM calculation (cached separately so it's fast on tab switch) ---
@st.cache_data
def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
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

@st.cache_data
def run_clustering(rfm_raw, n_clusters, winsorise=True):
    rfm = rfm_raw.copy()
    if winsorise:
        for col in ["Recency", "Frequency", "Monetary"]:
            upper = rfm[col].quantile(0.99)
            lower = rfm[col].quantile(0.01)
            rfm[col] = rfm[col].clip(lower=lower, upper=upper)
    rfm, X = transform_rfm(rfm)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=10)
    rfm["Cluster"] = km.fit_predict(X)
    sil = silhouette_score(X, rfm["Cluster"]) if len(rfm) > n_clusters else float("nan")
    return rfm, sil

@st.cache_data
def elbow_data(rfm_raw):
    _, X = transform_rfm(rfm_raw)
    max_k = min(8, len(X) - 1)
    inertias, silhouettes = [], []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return list(k_range), inertias, silhouettes

@st.cache_data
def build_cohort(df):
    df = df.copy()
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")
    first_purchase = df.groupby("CustomerID")["OrderMonth"].min().reset_index()
    first_purchase.columns = ["CustomerID", "CohortMonth"]
    df = df.merge(first_purchase, on="CustomerID")
    df["PeriodNumber"] = (df["OrderMonth"] - df["CohortMonth"]).apply(lambda x: x.n)
    cohort_counts = (
        df.groupby(["CohortMonth", "PeriodNumber"])["CustomerID"]
        .nunique()
        .reset_index()
        .pivot(index="CohortMonth", columns="PeriodNumber", values="CustomerID")
    )
    cohort_sizes = cohort_counts[0]
    retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
    retention.index = retention.index.astype(str)
    return retention, cohort_sizes.values

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_data()

# --- Sidebar ---
st.sidebar.title("Filters")
countries = ["All"] + sorted(df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)
if selected_country != "All":
    df = df[df["Country"] == selected_country]

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) == 2:
    df = df[
        (df["InvoiceDate"].dt.date >= date_range[0]) &
        (df["InvoiceDate"].dt.date <= date_range[1])
    ]

# --- Header ---
st.title("Customer Analytics Dashboard")
st.markdown(
    f"<p style='color:#6B7280;margin-top:-12px;font-size:0.95rem;'>"
    f"UCI Online Retail Dataset &nbsp;·&nbsp; "
    f"{df['InvoiceDate'].min().strftime('%b %Y')} - {df['InvoiceDate'].max().strftime('%b %Y')}"
    f"</p>",
    unsafe_allow_html=True,
)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧩 RFM Segmentation", "🔄 Cohort Retention"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Data quality summary ──────────────────────────────────────────────
    with st.expander("🧹 Data quality summary"):
        raw_count = load_raw_count()
        st.markdown(f"""
        | Check | Result |
        |---|---|
        | Raw rows | {raw_count:,} |
        | After cleaning | {len(df):,} |
        | Rows removed | {raw_count - len(df):,} ({(1 - len(df)/raw_count)*100:.1f}%) |
        | Unique customers | {df['CustomerID'].nunique():,} |
        | Date range | {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()} |
        | Guest checkout rows | {df['is_guest'].sum():,} ({df['is_guest'].mean()*100:.1f}%) |
        | Registered customer rows | {(~df['is_guest']).sum():,} |
        """)

    # ── KPI row ───────────────────────────────────────────────────────────
    guest_pct = df["is_guest"].mean() * 100
    st.info(
        f"Note: {guest_pct:.1f}% of transactions are guest checkouts (no CustomerID). "
        "Revenue figures include all transactions. RFM segmentation uses registered customers only."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{df['Revenue'].sum():,.0f}")
    col2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
    col3.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
    col4.metric("Avg Order Value", f"£{df.groupby('InvoiceNo')['Revenue'].sum().mean():,.2f}")

    st.divider()

    # ── Monthly revenue — area chart with spline smoothing ────────────────
    monthly_revenue = df.groupby("Month")["Revenue"].sum().reset_index()
    fig_line = go.Figure(go.Scatter(
        x=monthly_revenue["Month"],
        y=monthly_revenue["Revenue"],
        mode="lines",
        line=dict(color="#1D4ED8", width=2.5, shape="spline"),
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
            color_discrete_sequence=["#1D4ED8"],
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
            color_discrete_sequence=["#7C3AED"],
        )
        fig_prod.update_layout(yaxis=dict(categoryorder="total ascending"))
        fig_prod.update_xaxes(tickprefix="£", tickformat=",")
        st.plotly_chart(fig_prod, width='stretch')

    with st.expander("🔍 View raw data sample"):
        st.dataframe(df.head(500), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("RFM Customer Segmentation")
    st.markdown(
        "Customers are scored on **Recency** (days since last purchase), "
        "**Frequency** (number of orders), and **Monetary** (total spend). "
        "K-Means clustering groups them into actionable segments."
    )
    df_customers = df[~df["is_guest"]]
    rfm_raw = build_rfm(df_customers)
    max_k = min(8, len(rfm_raw) - 1)

    if max_k < 2:
        st.warning("Not enough customers in this selection for segmentation. Try 'All' countries or widen the date range.")
    else:
        # ── Elbow / silhouette chart ───────────────────────────────────────────
        with st.expander("📐 Choose number of clusters (Elbow method)", expanded=True):
            st.markdown("""
            Use these two charts together to choose the right number of customer segments (`k`):

            **Elbow Curve (left)** — shows how tightly packed the clusters are (inertia).
            Look for the "elbow": the point where the curve bends and the drop in inertia
            starts to flatten out. Adding more clusters beyond this point gives diminishing returns.

            **Silhouette Score (right)** — measures how well separated the clusters are (on a scale of 0 to 1).
            A higher score means customers within a segment are similar to each other and
            distinct from other segments. Look for a **local peak** — this is often the best `k`.

            > 💡 **Rule of thumb:** Find where the elbow & silhouette peak agree. If they differ,
            > favour the silhouette score. Also consider interpretability: 4 segments are usually
            > more actionable than 8.
            """)
            k_vals, inertias, silhouettes = elbow_data(rfm_raw)

            c1, c2 = st.columns(2)
            with c1:
                fig_elbow = go.Figure(go.Scatter(
                    x=k_vals, y=inertias, mode="lines+markers",
                    name="Inertia",
                    line=dict(color="#1D4ED8", width=2.5),
                    marker=dict(size=7),
                ))
                fig_elbow.update_layout(title="Elbow Curve (Inertia)", xaxis_title="k", yaxis_title="Inertia")
                st.plotly_chart(fig_elbow, width='stretch')
            with c2:
                fig_sil = go.Figure(go.Scatter(
                    x=k_vals, y=silhouettes, mode="lines+markers",
                    name="Silhouette",
                    line=dict(color="#059669", width=2.5),
                    marker=dict(size=7),
                ))
                fig_sil.update_layout(title="Silhouette Score by k", xaxis_title="k", yaxis_title="Score")
                st.plotly_chart(fig_sil, width='stretch')

        # ── k selector ────────────────────────────────────────────────────────
        default_k = min(4, max_k)
        n_clusters = st.select_slider(
            "Number of segments",
            options=list(range(2, max_k + 1)),
            value=default_k
        )
        winsorise = st.toggle("Winsorise outliers (clip at 1st/99th percentile)", value=True)
        rfm, sil = run_clustering(rfm_raw, n_clusters, winsorise=winsorise)
        rfm["Segment"] = assign_segment_labels(rfm)

        sil_str = f"{sil:.2f}" if not np.isnan(sil) else "n/a"
        st.caption(f"Silhouette score for k={n_clusters}: **{sil_str}** (higher = better defined clusters)")

        # ── Segment summary table ─────────────────────────────────────────────
        st.subheader("Segment Profiles")
        segment_summary = (
            rfm.groupby("Segment")
            .agg(
                Customers=("CustomerID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean"),
                Total_Revenue=("Monetary", "sum"),
            )
            .round(1)
            .reset_index()
            .sort_values("Avg_Monetary", ascending=False)
        )
        segment_summary.columns = ["Segment", "Customers", "Avg Recency (days)",
                                    "Avg Orders", "Avg Spend (£)", "Total Revenue (£)"]
        st.dataframe(segment_summary, width='stretch')

        # ── Scatter: Frequency vs Monetary, coloured by segment ───────────────
        st.subheader("Segment Visualisation")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_scatter = px.scatter(
                rfm, x="Frequency", y="Monetary", color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                hover_data=["CustomerID", "Recency"],
                title="Frequency vs Monetary by Segment",
                labels={"Monetary": "Total Spend (£)"},
                opacity=0.65,
            )
            fig_scatter.update_yaxes(tickprefix="£", tickformat=",")
            st.plotly_chart(fig_scatter, width='stretch')

        with col_b:
            fig_box = px.box(
                rfm, x="Segment", y="Monetary", color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                title="Spend Distribution by Segment",
                labels={"Monetary": "Total Spend (£)"},
            )
            fig_box.update_layout(showlegend=False)
            fig_box.update_yaxes(tickprefix="£", tickformat=",")
            st.plotly_chart(fig_box, width='stretch')

        # ── Radar chart: normalised RFM means per segment ─────────────────────
        st.subheader("Segment Radar Chart")
        radar_df = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()
        radar_norm = radar_df.copy()
        radar_norm["Recency"] = 1 - (radar_norm["Recency"] / radar_norm["Recency"].max())
        for col in ["Frequency", "Monetary"]:
            radar_norm[col] = radar_norm[col] / radar_norm[col].max()

        categories = ["Recency (inv.)", "Frequency", "Monetary"]
        fig_radar = go.Figure()
        for segment in radar_norm.index:
            vals = radar_norm.loc[segment].tolist()
            vals += vals[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=segment,
                line=dict(color=SEGMENT_COLORS.get(segment, "#1D4ED8"), width=2),
                opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB"),
                bgcolor="rgba(249,250,251,0.6)",
            ),
            title="Normalised RFM Profile per Segment",
        )
        st.plotly_chart(fig_radar, width='stretch')

        # ── Download ──────────────────────────────────────────────────────────
        st.subheader("Export")
        csv = rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]].to_csv(index=False)
        st.download_button(
            label="⬇️ Download segmented customers (CSV)",
            data=csv,
            file_name="rfm_segments.csv",
            mime="text/csv"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COHORT RETENTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Cohort Retention Analysis")
    st.markdown(
        "Each row is a cohort of customers defined by their first purchase month. "
        "Values show the percentage of that cohort who returned to purchase in each subsequent month."
    )

    df_cohort = df[~df["is_guest"]]  # registered customers only
    retention, cohort_sizes = build_cohort(df_cohort)

    fig_cohort = px.imshow(
        retention,
        text_auto=".0f",
        color_continuous_scale="Blues",
        labels={"x": "Months since first purchase", "y": "Cohort", "color": "Retention %"},
        title="Monthly Cohort Retention (%)",
        aspect="auto",
    )
    fig_cohort.update_xaxes(side="top")
    fig_cohort.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
    st.plotly_chart(fig_cohort, width='stretch')

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        avg_retention = retention.mean(axis=0).dropna()
        fig_avg = go.Figure(go.Scatter(
            x=avg_retention.index,
            y=avg_retention.values,
            mode="lines+markers",
            line=dict(color="#1D4ED8", width=2.5),
            marker=dict(size=7),
        ))
        fig_avg.update_layout(
            title="Average Retention Curve",
            xaxis_title="Months since first purchase",
            yaxis_title="Avg Retention (%)",
            showlegend=False,
        )
        st.plotly_chart(fig_avg, width='stretch')

    with col_r2:
        cohort_df = pd.DataFrame({
            "Cohort": retention.index,
            "Cohort Size": cohort_sizes.astype(int),
        })
        fig_sizes = px.bar(
            cohort_df, x="Cohort", y="Cohort Size",
            title="New Customers per Cohort",
            labels={"Cohort Size": "New Customers"},
            color_discrete_sequence=["#7C3AED"],
        )
        st.plotly_chart(fig_sizes, width='stretch')
