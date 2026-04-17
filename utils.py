import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import boxcox

pio.templates["portfolio"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(232,230,220,0.35)",
        title=dict(font=dict(size=14, color="#141413"), x=0, xanchor="left", pad=dict(b=12)),
        margin=dict(t=52, l=12, r=12, b=12),
        colorway=[
            "#B85F3D",
            "#2E7D68",
            "#7A52B3",
            "#2C78B7",
            "#B6861E",
            "#433D37"
        ],
        legend=dict(bgcolor="rgba(250,249,245,0.9)", bordercolor="#CEC9BC", borderwidth=1),
    )
)
pio.templates.default = "plotly+portfolio"

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
def load_data():
    df = pd.read_excel("data/online_retail.xlsx", engine="openpyxl")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    df["Country"] = df["Country"].replace({"EIRE": "Ireland"})

    noise_terms = ["POSTAGE", "DOTCOM", "BANK CHARGES", "MANUAL",
                   "AMAZONFEE", "CRUK", "SAMPLES", "TEST"]
    pattern = "|".join(noise_terms)
    df = df[~df["Description"].str.upper().str.contains(pattern, na=False)]
    df = df.dropna(subset=["Description"])

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

    df["is_guest"] = df["CustomerID"].isna()
    df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")

    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] >= 0.01]
    df = df.drop_duplicates()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    return df


@st.cache_data
def load_raw_count():
    raw = pd.read_excel("data/online_retail.xlsx", engine="openpyxl", usecols=[0])
    return len(raw)


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


@st.cache_data
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
        df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )

    summary = rfm_summary(
        invoices,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=obs_end,
        time_unit="W",
    )

    return summary.reset_index(drop=True)


def apply_sidebar_filters(df):
    st.logo("static/jordan_cheney_logo_new.png", size="large")
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
        help="Filter transactions by invoice date. RFM and cohort analyses will update accordingly."
    )
    if len(date_range) == 2:
        df = df[
            (df["InvoiceDate"].dt.date >= date_range[0]) &
            (df["InvoiceDate"].dt.date <= date_range[1])
        ]
    return df
