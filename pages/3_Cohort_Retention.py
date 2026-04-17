import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, apply_sidebar_filters, build_cohort

st.set_page_config(
    page_title="Cohort Retention · Customer Analytics",
    page_icon="🔄",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

st.header("Cohort Retention Analysis")
st.markdown(
    "Each row is a cohort of customers defined by their first purchase month. "
    "Values show the percentage of that cohort who returned to purchase in each subsequent month."
)

df_cohort = df[~df["is_guest"]]
retention, cohort_sizes = build_cohort(df_cohort)

fig_cohort = px.imshow(
    retention,
    text_auto=".0f",
    color_continuous_scale="Blues",
    labels={"x": "Months since first purchase", "y": "Cohort", "color": "Retention %"},
    title="Monthly Cohort Retention (%)",
    aspect="auto",
)
fig_cohort.update_xaxes(side="top")
fig_cohort.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
st.plotly_chart(fig_cohort, width='stretch')

col_r1, col_r2 = st.columns(2)
with col_r1:
    avg_retention = retention.mean(axis=0).dropna()
    fig_avg = go.Figure(go.Scatter(
        x=avg_retention.index,
        y=avg_retention.values,
        mode="lines+markers",
        line=dict(color="#1D4ED8", width=2.5),
        marker=dict(size=7),
    ))
    fig_avg.update_layout(
        title="Average Retention Curve",
        xaxis_title="Months since first purchase",
        yaxis_title="Avg Retention (%)",
        showlegend=False,
    )
    st.plotly_chart(fig_avg, width='stretch')

with col_r2:
    cohort_df = pd.DataFrame({
        "Cohort": retention.index,
        "Cohort Size": cohort_sizes.astype(int),
    })
    fig_sizes = px.bar(
        cohort_df, x="Cohort", y="Cohort Size",
        title="New Customers per Cohort",
        labels={"Cohort Size": "New Customers"},
        color_discrete_sequence=["#7C3AED"],
    )
    st.plotly_chart(fig_sizes, width='stretch')
