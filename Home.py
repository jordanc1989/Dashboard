import streamlit as st
from utils import load_data, apply_sidebar_filters

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

st.title("Customer Analytics Dashboard")
st.markdown(
    f"<p style='color:#6B7280;margin-top:-12px;font-size:0.95rem;'>"
    f"UCI Online Retail Dataset &nbsp;·&nbsp; "
    f"{df['InvoiceDate'].min().strftime('%b %Y')} - {df['InvoiceDate'].max().strftime('%b %Y')}"
    f"</p>",
    unsafe_allow_html=True,
)

st.markdown("""
Use the sidebar to navigate between pages and filter data by country or date range.

| Page | Description |
|---|---|
| 📈 Overview | Revenue trends, top countries & products, KPIs |
| 🧩 RFM Segmentation | K-Means customer segmentation using Recency, Frequency & Monetary value |
| 🔄 Cohort Retention | Monthly cohort retention heatmap and curves |
""")
