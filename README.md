# Customer Analytics Dashboard

Interactive Streamlit dashboard for customer behavior analytics using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

The app includes:
- sales and product performance views
- RFM customer segmentation
- churn propensity modelling
- probabilistic CLV prediction
- revenue time-series forecasting

## Features

- **Overview**: KPI cards, monthly trends and top countries/products by revenue
- **RFM Segmentation**: K-Means clustering with elbow/silhouette previews and segment labels
- **Churn Prediction**: random forest classification with threshold tuning, confusion matrix and precision-recall / ROC curves
- **CLV Prediction**: BG/NBD + Gamma-Gamma modelling via `pymc-marketing`
- **Revenue Forecasting**: SARIMA and Theta-method models (`statsmodels`) with holdout backtest metrics and 90% confidence intervals

## To do

- Enhance segment labelling (currently a static list chosen in order)


## Setup

### 1) Install dependencies

Recommended (uses the lock file for reproducible installs):

```bash
uv sync
```

Or with a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
├── app.py
├── app_pages/
│   ├── 1_Overview.py
│   ├── 2_RFM_Segmentation.py
│   ├── 3_Churn_Prediction.py
│   ├── 4_CLV_Prediction.py
│   └── 5_Revenue_Forecasting.py
├── utils.py
├── requirements.txt
├── static/
├── .streamlit/
│   ├── config.toml
│   └── pages.toml
└── data/
    └── online_retail_II.csv
```

## Dependencies

- `streamlit` for the app UI
- `pandas`, `numpy` for data wrangling
- `plotly` for interactive visualisations
- `scikit-learn`, `scipy` for RFM clustering and transformations
- `statsmodels` for SARIMA and Theta-method forecasting
- `pymc-marketing`, `pytensor`, `nutpie` for CLV modelling

## Notes

- The app currently reads `data/online_retail_II.csv` directly (in `utils.py`).
- If dependency installation fails, ensure `requirements.txt` includes all packages imported by the code (notably `numpy` and `scipy`).
