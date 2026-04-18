import pytensor
import streamlit as st

from utils import apply_sidebar_filters, load_data, render_dataset_subtitle

pytensor.config.cxx = ""  # Disables C compilation entirely - to fix current MacOS bug until Pytensor is updated

st.set_page_config(
    page_title="Customer analytics dashboard",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

st.logo("static/jordan_cheney_logo_new.png", size="large")

PAGE_CARDS = [
    (
        ":material/analytics:",
        "app_pages/1_Overview.py",
        "Overview",
        "Revenue trends, top countries and products, and headline KPIs.",
        "Exploration",
        "blue",
    ),
    (
        ":material/hub:",
        "app_pages/2_RFM_Segmentation.py",
        "RFM segmentation",
        "K-means customer groups from recency, frequency, and monetary value.",
        "Segmentation",
        "violet",
    ),
    (
        ":material/trending_down:",
        "app_pages/3_Churn_Prediction.py",
        "Churn prediction",
        "Random forest churn risk from behavioral features.",
        "Classification",
        "orange",
    ),
    (
        ":material/payments:",
        "app_pages/4_CLV_Prediction.py",
        "CLV prediction",
        "Customer lifetime value estimates and retention-style curves.",
        "CLV",
        "green",
    ),
    (
        ":material/show_chart:",
        "app_pages/5_Revenue_Forecasting.py",
        "Revenue forecasting",
        "SARIMA and Theta forecasts of future revenue.",
        "Forecasting",
        "gray",
    ),
]


def home_page():
    df = load_data()
    df = apply_sidebar_filters(df)

    render_dataset_subtitle(df)
    
    st.caption(
        "Use the sidebar to switch pages and filter by country or invoice date range."
    )
    st.space("medium")

    for row in range(0, len(PAGE_CARDS), 2):
        col_left, col_right = st.columns(2)
        for col, j in ((col_left, 0), (col_right, 1)):
            idx = row + j
            if idx >= len(PAGE_CARDS):
                break
            icon, page_path, title, desc, badge, badge_color = PAGE_CARDS[idx]
            with col:
                with st.container(border=True):
                    st.badge(badge, color=badge_color)
                    st.page_link(page_path, label=title, icon=icon)
                    st.caption(desc)


navigation = st.navigation(
    [
        st.Page(home_page, title="Home", icon=":material/home:"),
        st.Page("app_pages/1_Overview.py", title="Overview", icon=":material/analytics:"),
        st.Page("app_pages/2_RFM_Segmentation.py", title="RFM segmentation", icon=":material/hub:"),
        st.Page("app_pages/3_Churn_Prediction.py", title="Churn prediction", icon=":material/trending_down:"),
        st.Page("app_pages/4_CLV_Prediction.py", title="CLV prediction", icon=":material/payments:"),
        st.Page("app_pages/5_Revenue_Forecasting.py", title="Revenue forecasting", icon=":material/show_chart:"),
    ],
    position="sidebar",
)

page_title = "Customer analytics dashboard" if navigation.title == "Home" else navigation.title
st.title(page_title)

navigation.run()
