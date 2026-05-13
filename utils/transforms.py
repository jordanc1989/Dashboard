import streamlit as st
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

from utils.data import load_cancels


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
