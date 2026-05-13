import streamlit as st
import pandas as pd

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
    if isinstance(out["Country"].dtype, pd.CategoricalDtype):
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
