# Customer Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn&logoColor=white)
![pymc-marketing](https://img.shields.io/badge/pymc--marketing-0.19+-2980B9)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Interactive Streamlit dashboard for customer behaviour analytics using the [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

The app includes:
- sales and product performance views
- RFM customer segmentation
- churn propensity modelling
- probabilistic CLV prediction
- revenue time-series forecasting

## Features

- **Overview**: KPI cards, monthly trends and top countries/products by revenue
- **RFM Segmentation**: K-Means baseline with elbow/silhouette previews and R/F/M-derived segment labels, plus a Compare tab benchmarking Gaussian mixture, Agglomerative and HDBSCAN side by side
- **Churn Prediction**: random forest classification (5-fold stratified CV) with threshold tuning, confusion matrix and precision-recall / ROC curves
- **CLV Prediction**: BG/NBD + Gamma-Gamma modelling via `pymc-marketing`
- **Revenue Forecasting**: SARIMA, Theta, ETS (Holt-Winters) and a seasonal-naive model (`statsmodels`) with holdout backtest metrics and configurable prediction intervals. Every model is scored against the seasonal-naive baseline ("same period last year"), so the backtest shows whether it actually beats the obvious benchmark. Short backtest windows (< 2 seasonal cycles) reconstruct seasonality from training data only, so the holdout forecast stays seasonal without look-ahead

## Setup

Requires **Python 3.13+**.

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build the data artifacts

The app reads precomputed Parquet files rather than the raw CSV (see
[Data pipeline](#data-pipeline)). Generate them once from the source CSV:

```bash
python scripts/build_dataset.py
```

This writes `data/online_retail_clean.parquet`, `data/online_retail_cancels.parquet`
and `data/dataset_meta.json`. Re-run it whenever the raw CSV or the cleaning
logic in `utils/data.py` changes.

### 3) Run the app

```bash
streamlit run app.py
```

## Project Structure

```text
.
├── app.py                      # entry point; defines st.navigation
├── app_pages/
│   ├── 1_Overview.py
│   ├── 2_RFM_Segmentation.py
│   ├── 3_Churn_Prediction.py
│   ├── 4_CLV_Prediction.py
│   └── 5_Revenue_Forecasting.py
├── utils/                      # shared package
│   ├── data.py                 # loading, cleaning and Parquet artifacts
│   ├── transforms.py           # RFM, cohort, churn and CLV feature builds
│   ├── clustering.py           # segmentation algorithms and metrics
│   ├── theme.py                # Plotly template and colour palette
│   └── ui.py                   # page chrome, headers, sidebar filters
├── scripts/
│   └── build_dataset.py        # precompute cleaned Parquet artifacts from the CSV
├── .streamlit/
│   ├── config.toml             # theme and server config
│   └── pages.toml
├── data/
│   ├── online_retail_II.csv            # raw source (build input only)
│   ├── online_retail_clean.parquet     # cleaned orders (read at runtime)
│   ├── online_retail_cancels.parquet   # cancel/return invoices
│   └── dataset_meta.json               # raw row count for data-quality stats
├── static/                     # fonts, logos, favicon
├── requirements.txt
└── LICENSE
```

## Dependencies

- `streamlit` for the app UI
- `pandas`, `numpy`, `pyarrow` for data wrangling and Parquet I/O
- `plotly` for interactive visualisations
- `scikit-learn`, `scipy` for clustering and transformations
- `statsmodels` for SARIMA and Theta-method forecasting
- `pymc-marketing` (with its `pytensor` backend) for CLV modelling

See `requirements.txt` for pinned version constraints.

## Data pipeline

The raw CSV is ~90 MB / 1.07M rows. Parsing and cleaning it on every cold start
peaks at ~1 GB RSS, which exceeds the Streamlit Community Cloud memory limit and
causes OOM restarts. To avoid this, the cleaning runs once, offline, via
`scripts/build_dataset.py`:

- `utils/data.py` holds the cleaning logic (`_build_clean_orders`, `_build_cancels`).
- The build script writes compact Parquet artifacts (~7 MB total).
- At runtime, `load_data()` / `load_cancels()` read those Parquet files (with a
  CSV fallback if they are missing), so the app never re-parses the CSV.

This drops the cold-start peak memory from ~1 GB to a few hundred MB and the
resident dataset from ~125 MB to ~80 MB (the heavily repeated `Description`
column is stored as a `category`).

## Notes

- At runtime the app reads `data/online_retail_clean.parquet` (in `utils/data.py`); reads are cached with `@st.cache_data`. The raw `online_retail_II.csv` is only needed by the build script.
- Navigation is defined programmatically via `st.navigation` in `app.py`.

## License

Released under the [MIT License](LICENSE).
