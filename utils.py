import streamlit as st
import pandas as pd
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

# Brand-aligned sequential scales for heatmaps (avoid default Blues / RdYlGn only)
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
        plot_bgcolor="rgba(232,230,220,0.35)",
        font=dict(family="SpaceGrotesk", size=12, color="#3d3a2a"),
        title=dict(
            font=dict(size=15, color="#141413", family="SpaceGrotesk"),
            x=0,
            xanchor="left",
            pad=dict(b=12),
        ),
        margin=dict(t=52, l=12, r=12, b=12),
        colorway=list(CHART_COLORWAY),
        legend=dict(bgcolor="rgba(250,249,245,0.9)", bordercolor="#CEC9BC", borderwidth=1),
    )
)
pio.templates.default = "plotly+portfolio"

# Warm neutrals for reference lines and grid chrome (matches theme, not cool Tailwind grays)
NEUTRAL_GRID = "#948A78"
NEUTRAL_RADAR_GRID = "#CEC9BC"


def finalize_fig(fig, *, unified_hover: bool = False, uirevision: str | None = None):
    """Apply shared Plotly layout so all charts use the portfolio template consistently."""
    kwargs: dict = {"template": "plotly+portfolio"}
    if unified_hover:
        kwargs["hovermode"] = "x unified"
    if uirevision is not None:
        kwargs["uirevision"] = uirevision
    fig.update_layout(**kwargs)
    return fig


SEGMENT_COLORS = {
    "Champions":           "#B85F3D",
    "Loyal Customers":     "#2E7D68",
    "Potential Loyalists": "#7A52B3",
    "Promising":           "#2C78B7",
    "Need Attention":      "#B6861E",
    "At Risk":             "#433D37"
}

SEGMENT_LABELS = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Promising",
    "Need Attention",
    "At Risk"
]


def render_dataset_subtitle(df: pd.DataFrame) -> None:
    """Standard date-range line under page titles (matches Streamlit body tone)."""
    st.caption(
        f"UCI Online Retail Dataset · "
        f"{df['InvoiceDate'].min():%b %Y} – {df['InvoiceDate'].max():%b %Y}"
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
    label_map = {cid: SEGMENT_LABELS[rank] for rank, cid in enumerate(ranked)}
    return rfm["Cluster"].map(label_map)


@st.cache_data
def _load_raw():
    """Single source of truth for the raw CSV read.

    Everything downstream (orders, cancels, raw counts) derives from this
    cached frame to avoid re-reading the file three times.
    """
    df = pd.read_csv("data/online_retail_II.csv")
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


@st.cache_data
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
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    return df


@st.cache_data
def load_raw_count():
    return len(_load_raw())


@st.cache_data
def load_cancels():
    """Load only cancel (return) invoices, which `load_data` strips out.

    Uses `_retail_csv_line_filters` like `load_data` so cancel counts stay
    comparable. Returned columns: Customer ID, InvoiceDate, Invoice.
    """
    df = _retail_csv_line_filters(_load_raw())

    df = df[df["Invoice"].str.startswith("C")]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype("Int64").astype("string")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
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


@st.cache_data(max_entries=16)
def elbow_data(rfm_raw, winsorise=True, max_segments=6):
    rfm = rfm_raw.copy()

    if winsorise:
        for col in ["Recency", "Frequency", "Monetary"]:
            lower = rfm[col].quantile(0.01)
            upper = rfm[col].quantile(0.99)
            rfm[col] = rfm[col].clip(lower=lower, upper=upper)

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
    **churned = 1** if they made no purchase in `(cutoff, max_date]`, else 0.
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


@st.cache_data(max_entries=16)
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
    st.sidebar.markdown("### Filters")
    st.sidebar.divider()
    with st.sidebar.expander("Data scope", expanded=True):
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
