# Customer Analytics Dashboard

Interactive Streamlit dashboard for customer behavior analytics using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

The app includes:
- sales and product performance views,
- RFM customer segmentation,
- churn propensity modeling,
- probabilistic CLV prediction, and
- revenue time-series forecasting.

## Features

- **Overview**: KPI cards, monthly trends, and top countries/products by revenue
- **RFM Segmentation**: K-Means clustering with elbow/silhouette diagnostics and segment labels
- **Churn Prediction**: random forest classification with threshold tuning, confusion matrix, and precision-recall curves
- **CLV Prediction**: BG/NBD + Gamma-Gamma modeling via `pymc-marketing`
- **Revenue Forecasting**: SARIMA and Theta-method models (`statsmodels`) with holdout backtest metrics and 90% confidence intervals

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

## Project Structure

```text
.
├── app.py
├── app_pages/
│   ├── 1_Overview.py
│   ├── 2_RFM_Segmentation.py
│   ├── 3_Churn_Prediction.py
│   ├── 4_CLV_Prediction.py
│   └── 5_Revenue_Forecasting.py
├── utils.py
├── requirements.txt
└── data/
    └── online_retail_II.csv
```

## Dependencies

- `streamlit` for the app UI
- `pandas`, `numpy` for data wrangling
- `plotly` for interactive visualizations
- `scikit-learn`, `scipy` for RFM clustering and transformations
- `statsmodels` for SARIMA and Theta-method forecasting
- `pymc-marketing`, `pytensor`, `nutpie` for CLV modeling

## Notes

- The app currently reads `data/online_retail_II.csv` directly (in `utils.py`).
- If dependency installation fails, ensure `requirements.txt` includes all packages imported by the code (notably `numpy` and `scipy`).
