# Customer Analytics Dashboard

Interactive Streamlit dashboard for customer behavior analytics using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

The app includes:
- sales and product performance views,
- RFM customer segmentation,
- cohort retention analysis, and
- probabilistic CLV prediction.

## Features

- **Overview**: KPI cards, monthly trends, and top countries/products by revenue
- **RFM Segmentation**: K-Means clustering with elbow/silhouette diagnostics and segment labels
- **Cohort Retention**: month-on-month retention heatmap and retention curves
- **CLV Prediction**: BG/NBD + Gamma-Gamma modeling via `pymc-marketing`

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

### 3) Add the dataset

Place the Excel file at:

`data/online_retail.xlsx`

### 4) Run the app

```bash
streamlit run Home.py
```

## Project Structure

```text
.
├── Home.py
├── pages/
│   ├── 1_Overview.py
│   ├── 2_RFM_Segmentation.py
│   ├── 3_Cohort_Retention.py
│   └── 4_CLV_Prediction.py
├── utils.py
├── requirements.txt
└── data/
    └── online_retail.xlsx
```

## Dependencies

- `streamlit` for the app UI
- `pandas`, `numpy` for data wrangling
- `plotly` for interactive visualizations
- `scikit-learn`, `scipy` for RFM clustering and transformations
- `openpyxl` for reading the Excel dataset
- `pymc-marketing`, `pytensor` for CLV modeling
