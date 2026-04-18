import pytensor
import streamlit as st

from utils import apply_sidebar_filters, load_data, render_dataset_subtitle

pytensor.config.cxx = ""  # Disables C compilation entirely - to fix current MacOS bug until Pytensor is updated

st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

st.logo("static/jordan_cheney_logo_new.png", size="large")

PAGE_CARDS = [
    (
        ":material/analytics:",
        "app_pages/1_Overview.py",
        "Overview",
        "Revenue trends, top countries & products, KPIs",
        "Exploration",
    ),
    (
        ":material/hub:",
        "app_pages/2_RFM_Segmentation.py",
        "RFM Segmentation",
        "K-Means customer segmentation using Recency, Frequency & Monetary value",
        "Segmentation",
    ),
    (
        ":material/trending_down:",
        "app_pages/3_Churn_Prediction.py",
        "Churn Prediction",
        "Predicted customer churn using Random Forest",
        "Classification",
    ),
    (
        ":material/payments:",
        "app_pages/4_CLV_Prediction.py",
        "CLV Prediction",
        "Predicted customer lifetime value and retention curves",
        "CLV",
    ),
    (
        ":material/show_chart:",
        "app_pages/5_Revenue_Forecasting.py",
        "Revenue Forecasting",
        "Time-series forecasts of future revenue (SARIMA / Theta)",
        "Forecasting",
    ),
]


def home_page():
    df = load_data()
    df = apply_sidebar_filters(df)

    render_dataset_subtitle(df)

    st.markdown(
        "Use the sidebar to navigate between pages and filter data by country or date range."
    )

    for row in range(0, len(PAGE_CARDS), 2):
        col_left, col_right = st.columns(2)
        for col, j in ((col_left, 0), (col_right, 1)):
            idx = row + j
            if idx >= len(PAGE_CARDS):
                break
            icon, page_path, title, desc, badge = PAGE_CARDS[idx]
            with col:
                with st.container(border=True):
                    st.badge(badge)
                    st.page_link(page_path, label=title, icon=icon)
                    st.caption(desc)


navigation = st.navigation(
    [
        st.Page(home_page, title="Home", icon=":material/home:"),
        st.Page("app_pages/1_Overview.py", title="Overview", icon=":material/analytics:"),
        st.Page("app_pages/2_RFM_Segmentation.py", title="RFM Segmentation", icon=":material/hub:"),
        st.Page("app_pages/3_Churn_Prediction.py", title="Churn Prediction", icon=":material/trending_down:"),
        st.Page("app_pages/4_CLV_Prediction.py", title="CLV Prediction", icon=":material/payments:"),
        st.Page("app_pages/5_Revenue_Forecasting.py", title="Revenue Forecasting", icon=":material/show_chart:"),
    ],
    position="sidebar",
)

page_title = "Customer Analytics Dashboard" if navigation.title == "Home" else navigation.title
st.title(page_title)

navigation.run()
