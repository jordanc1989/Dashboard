import json
from pathlib import Path

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

# Precomputed artifacts. The app reads these at runtime so it never re-parses the
# 90 MB raw CSV or re-runs the cleaning pipeline on a cold start (that pipeline
# peaks at ~1 GB RSS and OOM-kills the app on Streamlit Community Cloud). Build
# them with `python scripts/build_dataset.py`; the CSV fallbacks below keep local
# dev working if the parquet files are missing.
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_RAW_CSV = _DATA_DIR / "online_retail_II.csv"
_CLEAN_PARQUET = _DATA_DIR / "online_retail_clean.parquet"
_CANCELS_PARQUET = _DATA_DIR / "online_retail_cancels.parquet"
_META_JSON = _DATA_DIR / "dataset_meta.json"


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
    out = out[~out["StockCode"].str.upper().str.startswith("POST", na=False)]
    return out


def _read_raw_csv():
    """Single source of truth for the raw CSV read.

    Keeps the high-cardinality identifier columns as the pandas ``string`` dtype
    rather than coercing to object via ``astype(str)`` - object strings cost
    several times more memory per row on pandas < 3.0.
    """
    df = pd.read_csv(
        _RAW_CSV,
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
    return df


def _build_clean_orders():
    """Full cleaning pipeline: raw CSV -> deduplicated, cancel-reconciled orders.

    This is the expensive path. At runtime ``load_data`` reads the precomputed
    parquet instead; this only runs in the build script or as a local fallback.
    """
    df = _retail_csv_line_filters(_read_raw_csv())

    cancel_mask = df["Invoice"].str.startswith("C", na=False)
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
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype("string")

    # Description repeats heavily across rows (~5k unique over ~1M rows) and is
    # only ever a display label, never a join/group key - storing it as a
    # category cuts its footprint from ~35 MB to ~2 MB.
    df["Description"] = df["Description"].astype("category")
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")

    return df.reset_index(drop=True)


def _build_cancels():
    """Cancel (return) invoices that ``_build_clean_orders`` strips out.

    Returned columns: Customer ID, InvoiceDate, Invoice.
    """
    df = _retail_csv_line_filters(_read_raw_csv())

    df = df[df["Invoice"].str.startswith("C", na=False)]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype("Int64").astype("string")
    df = df.dropna(subset=["InvoiceDate"])
    return df[["Customer ID", "InvoiceDate", "Invoice"]].reset_index(drop=True)


@st.cache_data(max_entries=1)
def load_data():
    """Cleaned, cancel-reconciled order lines.

    Prefers the precomputed parquet (fast, low peak memory); falls back to the
    full CSV pipeline if the artifact is missing.
    """
    if _CLEAN_PARQUET.exists():
        return pd.read_parquet(_CLEAN_PARQUET)
    return _build_clean_orders()


@st.cache_data(max_entries=1)
def load_cancels():
    """Cancel (return) invoices, parquet-first with a CSV fallback."""
    if _CANCELS_PARQUET.exists():
        return pd.read_parquet(_CANCELS_PARQUET)
    return _build_cancels()


@st.cache_data(max_entries=1)
def load_raw_count():
    """Row count of the raw CSV, used for the data-quality 'rows removed' stat.

    Read from the build metadata so the raw CSV need not ship with the app.
    """
    if _META_JSON.exists():
        return int(json.loads(_META_JSON.read_text())["raw_count"])
    return len(_read_raw_csv())
