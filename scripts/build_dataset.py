"""Precompute the cleaned dataset artifacts the dashboard reads at runtime.

Reading and cleaning the 90 MB raw CSV on every cold start peaks at ~1 GB RSS,
which OOM-kills the app on Streamlit Community Cloud. This script runs that work
once, offline, and writes compact parquet files (~7 MB) plus a small metadata
JSON. The app then just `read_parquet`s them, with a peak RSS in the low
hundreds of MB.

Run from the repo root:

    python scripts/build_dataset.py

Re-run whenever the raw CSV or the cleaning logic in `utils/data.py` changes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data import (  # noqa: E402
    _build_cancels,
    _build_clean_orders,
    _read_raw_csv,
    _CANCELS_PARQUET,
    _CLEAN_PARQUET,
    _META_JSON,
)


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> None:
    print("Reading raw CSV and counting rows...")
    raw_count = len(_read_raw_csv())

    print("Building cleaned orders...")
    orders = _build_clean_orders()
    orders.to_parquet(_CLEAN_PARQUET, index=False, compression="zstd")

    print("Building cancels...")
    cancels = _build_cancels()
    cancels.to_parquet(_CANCELS_PARQUET, index=False, compression="zstd")

    _META_JSON.write_text(json.dumps({"raw_count": int(raw_count)}, indent=2))

    print("\nDone.")
    print(f"  raw rows            : {raw_count:,}")
    print(f"  clean orders        : {len(orders):,} rows -> {_CLEAN_PARQUET.name} ({_mb(_CLEAN_PARQUET):.1f} MB)")
    print(f"  cancels             : {len(cancels):,} rows -> {_CANCELS_PARQUET.name} ({_mb(_CANCELS_PARQUET):.1f} MB)")
    print(f"  in-memory orders    : {orders.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  metadata            : {_META_JSON.name}")


if __name__ == "__main__":
    main()
