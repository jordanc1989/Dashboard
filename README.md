# Customer Analytics Dashboard

An interactive Streamlit dashboard for customer behaviour analysis using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail). It covers revenue trends, RFM-based customer segmentation, and cohort retention analysis.

## Features

- **Overview** — monthly revenue trends, top countries and products by revenue
- **RFM Segmentation** — K-Means clustering with elbow/silhouette charts to guide `k` selection; customers labelled into segments (Champions, At Risk, etc.)
- **Cohort Retention** — heatmap showing month-on-month retention by acquisition cohort

## Setup

**1. Clone the repo and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Run the app**

```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── data/
    └── online_retail.xlsx  # Dataset
```

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` / `numpy` | Data wrangling |
| `plotly` | Interactive charts |
| `scikit-learn` | K-Means clustering |
| `scipy` | Box-Cox transformation for RFM normalisation |
| `openpyxl` | Reading `.xlsx` files |
