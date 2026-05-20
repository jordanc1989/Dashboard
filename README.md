# Customer Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn&logoColor=white)
![pymc-marketing](https://img.shields.io/badge/pymc--marketing-0.19+-2980B9)
![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9?logo=uv&logoColor=white)
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
- **Revenue Forecasting**: SARIMA and Theta-method models (`statsmodels`) with holdout backtest metrics and 90% confidence intervals

## Setup

Requires **Python 3.14+**.

### 1) Install dependencies

Recommended (uses the lock file for reproducible installs):

```bash
uv sync
```


### 2) Run the app

```bash
uv run streamlit run app.py
# or, if using a venv:
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
│   ├── data.py                 # CSV loading and cleaning
│   ├── transforms.py           # RFM, cohort, churn and CLV feature builds
│   ├── clustering.py           # segmentation algorithms and metrics
│   ├── theme.py                # Plotly template and colour palette
│   └── ui.py                   # page chrome, headers, sidebar filters
├── .streamlit/
│   ├── config.toml             # theme and server config
│   └── pages.toml
├── data/
│   └── online_retail_II.csv
├── static/                     # fonts, logos, favicon
├── pyproject.toml
├── uv.lock
└── LICENSE
```

## Dependencies

- `streamlit` for the app UI
- `pandas`, `numpy` for data wrangling
- `plotly` for interactive visualisations
- `scikit-learn`, `scipy` for clustering and transformations
- `statsmodels` for SARIMA and Theta-method forecasting
- `pymc-marketing` (with its `pytensor` backend) for CLV modelling

See `pyproject.toml` for pinned version constraints and `uv.lock` for the fully resolved set.

## Notes

- The app reads `data/online_retail_II.csv` directly (in `utils/data.py`); the read is cached with `@st.cache_data`.
- Navigation is defined programmatically via `st.navigation` in `app.py`.

## License

Released under the [MIT License](LICENSE).
