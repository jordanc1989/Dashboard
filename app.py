import streamlit as st

from utils import apply_sidebar_filters, load_data

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

def home_page():
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
| 🔄 Churn Prediction | Predicted customer churn using Random Forest |
| 💰 CLV Prediction | Predicted customer lifetime value and retention curves |
| 🔮 Revenue Forecasting | Time-series forecasts of future revenue (SARIMA / Theta) |
""")


navigation = st.navigation(
    [
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page("pages/1_Overview.py", title="Overview", icon="📈"),
        st.Page("pages/2_RFM_Segmentation.py", title="RFM Segmentation", icon="🧩"),
        st.Page("pages/3_Churn_Prediction.py", title="Churn Prediction", icon="🔄"),
        st.Page("pages/4_CLV_Prediction.py", title="CLV Prediction", icon="💰"),
        st.Page("pages/5_Revenue_Forecasting.py", title="Revenue Forecasting", icon="🔮"),
    ],
    position="sidebar",
)
navigation.run()
