import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    CHART_COLORWAY,
    load_data,
    apply_sidebar_filters,
    build_rfm,
    run_clustering,
    elbow_data,
    assign_segment_labels,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    render_page_header,
    section,
    NEUTRAL_RADAR_GRID,
    finalise_fig,
)

st.set_page_config(
    page_title="RFM segmentation · Customer analytics",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

render_page_header("rfm", df)

df_customers = df[~df["is_guest"]]
rfm_raw = build_rfm(df_customers)
max_k = min(len(SEGMENT_LABELS), len(rfm_raw) - 1)

if max_k < 2:
    st.warning(
        "Not enough customers in this selection for segmentation. "
        "Try 'All' countries or widen the date range."
    )
else:
    default_k = min(4, max_k)

    section("Clustering controls", eyebrow="Model inputs")
    with st.container(border=True):
        ctrl_cols = st.columns([2, 3])
        with ctrl_cols[0]:
            winsorise = st.toggle(
                "Winsorise outliers",
                value=True,
                help="Clip Recency / Frequency / Monetary at the 1st and 99th "
                "percentiles before fitting K-means. Helps with long-tailed spend.",
            )
        with ctrl_cols[1]:
            st.caption(
                "Winsorising reduces the influence of extreme outliers so clusters "
                "reflect the broader customer base rather than a handful of whales."
            )

    # ── Elbow / silhouette chart ──────────────
    with st.expander(
        "Choose number of clusters (Elbow method)",
        expanded=True,
        icon=":material/insights:",
    ):
        st.markdown("""
        Use these two charts together to choose the right number of customer segments (`k`):

        **Elbow Curve:** shows how tightly packed the clusters are (inertia).
        Look for the "elbow", the point where the curve bends and the drop 
        starts to flatten out. Adding more clusters beyond this point gives diminishing returns.

        **Silhouette Score:** measures how well separated the clusters are (on a scale of 0 to 1).
        A higher score means customers within a segment are more similar to each other & more
        distinct from other segments. **Local peaks** are often the best `k`.

        > **Tip:** Find where the elbow and silhouette peak agree. If they differ,
        > favour the silhouette score. Consider interpretability: 3-4 segments are usually
        > more actionable for the business than 6.
        """)

        k_vals, inertias, silhouettes = elbow_data(
            rfm_raw,
            winsorise=winsorise,
            max_segments=len(SEGMENT_LABELS)
        )

        c1, c2 = st.columns(2)

        with c1:
            fig_elbow = go.Figure(go.Scatter(
                x=k_vals,
                y=inertias,
                mode="lines+markers",
                name="Inertia",
                line=dict(color=CHART_COLORWAY[3], width=2.5),
                marker=dict(size=7),
            ))
            fig_elbow.update_layout(
                title="Elbow curve (inertia)",
                xaxis_title="k",
                yaxis_title="Inertia"
            )
            finalise_fig(fig_elbow)
            st.plotly_chart(fig_elbow, width="stretch")

        with c2:
            fig_sil = go.Figure(go.Scatter(
                x=k_vals,
                y=silhouettes,
                mode="lines+markers",
                name="Silhouette",
                line=dict(color=CHART_COLORWAY[1], width=2.5),
                marker=dict(size=7),
            ))
            fig_sil.update_layout(
                title="Silhouette score by k",
                xaxis_title="k",
                yaxis_title="Score"
            )
            finalise_fig(fig_sil)
            st.plotly_chart(fig_sil, width="stretch")

    # ── k selector ──────────────────────────────────────────────────────
    st.space("small")
    with st.container(border=True):
        n_clusters = st.select_slider(
            "Number of segments (k)",
            options=list(range(2, max_k + 1)),
            value=default_k,
            label_visibility='visible'
        )

    rfm, sil = run_clustering(rfm_raw, n_clusters, winsorise=winsorise)
    rfm["Segment"] = assign_segment_labels(rfm)
    rfm["Segment"] = pd.Categorical(
        rfm["Segment"],
        categories=SEGMENT_LABELS[:n_clusters],
        ordered=True
    )

    sil_str = f"{sil:.2f}" if not np.isnan(sil) else "n/a"
    st.caption(
        f"Silhouette score for k={n_clusters}: **{sil_str}** "
        "(higher = better defined clusters)"
    )

    # ── Segment summary table ───────────────────────────────────────────
    st.space("small")
    section("Segment profiles", eyebrow="Customer mix")
    segment_summary = (
        rfm.groupby("Segment", observed=False)
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(1)
        .reset_index()
    )

    segment_summary.columns = [
        "Segment",
        "Customers",
        "Avg Recency (days)",
        "Avg Orders",
        "Avg Spend (£)",
        "Total Revenue (£)"
    ]
    st.dataframe(
        segment_summary,
        width="stretch",
        column_config={
            "Customers": st.column_config.NumberColumn(format="%,d"),
            "Avg Recency (days)": st.column_config.NumberColumn(format="%.1f"),
            "Avg Orders": st.column_config.NumberColumn(format="%.1f"),
            "Avg Spend (£)": st.column_config.NumberColumn(format="£ %.2f"),
            "Total Revenue (£)": st.column_config.NumberColumn(format="£ %,.0f"),
        },
        hide_index=True)

    # ── Scatter: Frequency vs Monetary, coloured by segment ────
    st.space("small")
    section("Segment visualisation", eyebrow="Distributions")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Segment": list(SEGMENT_LABELS[:n_clusters])},
            hover_data=["Customer ID", "Recency"],
            title="Frequency vs monetary by segment",
            labels={"Monetary": "Total Spend (£)"},
            opacity=0.65,
        )
        fig_scatter.update_yaxes(tickprefix="£", tickformat=",")
        finalise_fig(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")

    with col_b:
        fig_box = px.box(
            rfm,
            x="Segment",
            y="Monetary",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Segment": list(SEGMENT_LABELS[:n_clusters])},
            title="Spend distribution by segment",
            labels={"Monetary": "Total Spend (£)"},
        )
        fig_box.update_layout(showlegend=False)
        fig_box.update_yaxes(tickprefix="£", tickformat=",")
        finalise_fig(fig_box)
        st.plotly_chart(fig_box, width="stretch")

    # ── Radar chart: normalised RFM means per segment ─────────
    st.space("medium")
    section("Segment radar chart", eyebrow="Normalised RFM")
    radar_df = (
        rfm.groupby("Segment", observed=False)[["Recency", "Frequency", "Monetary"]]
        .mean()
        .dropna()
    )

    radar_norm = radar_df.copy()
    radar_norm["Recency"] = 1 - (radar_norm["Recency"] / radar_norm["Recency"].max())

    for col in ["Frequency", "Monetary"]:
        radar_norm[col] = radar_norm[col] / radar_norm[col].max()

    categories = ["Recency (inv.)", "Frequency", "Monetary"]
    fig_radar = go.Figure()

    for segment in radar_norm.index:
        vals = radar_norm.loc[segment].tolist()
        vals += vals[:1]

        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=str(segment),
            line=dict(color=SEGMENT_COLORS.get(str(segment), CHART_COLORWAY[0]), width=2),
            opacity=0.8,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=NEUTRAL_RADAR_GRID),
            angularaxis=dict(gridcolor=NEUTRAL_RADAR_GRID),
            bgcolor="rgba(253,253,248,0.55)",
        ),
        title="Normalised RFM profile per segment",
    )
    finalise_fig(fig_radar)
    st.plotly_chart(fig_radar, width="stretch")

    # ── Download ─────────
    st.space("small")
    section("Export", eyebrow="Download results")
    csv = rfm[
        ["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]
    ].to_csv(index=False)

    st.download_button(
        label="Download segmented customers (CSV)",
        data=csv,
        file_name="rfm_segments.csv",
        mime="text/csv",
        icon=":material/download:",
    )
