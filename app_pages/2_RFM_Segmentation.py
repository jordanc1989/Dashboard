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
    MAX_SEGMENTS,
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
max_k = min(MAX_SEGMENTS, len(rfm_raw) - 1)

if max_k < 2:
    st.warning(
        "Not enough customers in this selection for segmentation. "
        "Try 'All' countries or widen the date range."
    )
else:
    default_k = min(4, max_k)

    # ── k selector + silhouette ─────────────────────────────────────────
    st.space("small")
    with st.container(border=True):
        ctrl_cols = st.columns([3, 1, 1])
        with ctrl_cols[0]:
            n_clusters = st.select_slider(
                "Number of segments (k)",
                options=list(range(2, max_k + 1)),
                value=default_k,
                label_visibility='visible'
            )
        with ctrl_cols[1]:
            winsorise_pct = st.slider(
                "Winsorise percentile",
                min_value=95,
                max_value=100,
                value=99,
                step=1,
                help="Clip Recency / Frequency / Monetary at this percentile (and its "
                "mirror) before fitting. Set to 100 to disable winsorisation entirely.",
            )
        sil_slot = ctrl_cols[2].empty()

    # ── Elbow / silhouette chart ──────────────
    with st.expander(
        "Choose number of clusters (Elbow method)",
        expanded=False,
        icon=":material/insights:",
    ):
        st.markdown("""
        Use these two charts together to choose the right number of customer segments (`k`):

        **Elbow Curve:** shows how tightly packed the clusters are (inertia).
        Look for the "elbow", the point where the curve bends and the drop 
        starts to flatten out. Adding more clusters beyond this point gives diminishing returns.

        **Silhouette Score:** measures how well separated the clusters are (on a scale of −1 to 1).
        A higher score means customers within a segment are more similar to each other & more
        distinct from other segments. **Local peaks** are often the best `k`.

        > **Tip:** Find where the elbow and silhouette peak agree. If they differ,
        > favour the silhouette score. Remember that 3 or 4 segments are usually
        > more actionable for a business than 6.
        """)

        k_vals, inertias, silhouettes = elbow_data(
            rfm_raw,
            winsorise_pct=winsorise_pct,
            max_segments=MAX_SEGMENTS
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

    rfm, sil = run_clustering(rfm_raw, n_clusters, winsorise_pct=winsorise_pct)
    rfm["Segment"], seg_colors, ordered_labels = assign_segment_labels(rfm)

    # Merge segment labels onto raw values; winsorized frame only served clustering.
    rfm_display = rfm_raw.merge(rfm[["Customer ID", "Segment"]], on="Customer ID")
    rfm_display["Segment"] = pd.Categorical(
        rfm_display["Segment"],
        categories=[cat for cat in ordered_labels if cat in rfm_display["Segment"].values],
        ordered=True
    )

    sil_str = f"{sil:.2f}" if sil is not None and not np.isnan(sil) else "n/a"
    sil_slot.metric(
        "Silhouette score",
        sil_str,
        help="Measures how well-separated the clusters are (−1 to 1). Higher is better.",
    )

    # ── Segment summary table ───────────────────────────────────────────
    st.space("small")
    section("Segment profiles", eyebrow="Customer mix")
    segment_summary = (
        rfm_display.groupby("Segment", observed=False)
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

    # ── Scatter: full width ────────────────────────────────────────────────
    st.space("small")
    section("Segment visualisation", eyebrow="Breakdown")
    use_3d = st.toggle("3D view", value=False, key="scatter_3d")

    if use_3d:
        fig_scatter = px.scatter_3d(
            rfm_display,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            color_discrete_map=seg_colors,
            category_orders={"Segment": ordered_labels},
            hover_data=["Customer ID"],
            title="RFM: Three Dimensional",
            labels={
                "Recency": "Days since last purchase",
                "Frequency": "Orders",
                "Monetary": "Total Spend",
            },
            opacity=0.7,
        )
        fig_scatter.update_traces(marker_size=2)
        fig_scatter.update_layout(
            scene=dict(
                xaxis=dict(autorange="reversed"),
                zaxis=dict(
                    type="log",
                    tickvals=[1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
                    ticktext=["£1", "£10", "£100", "£1,000", "£10,000", "£100,000", "£1,000,000"],
                ),
            ),
            height=700,
            margin=dict(t=40, l=0, r=0, b=0),
            legend=dict(orientation="h", y=-0.05),
        )
    else:
        fig_scatter = px.scatter(
            rfm_display,
            x="Recency",
            y="Monetary",
            color="Segment",
            color_discrete_map=seg_colors,
            category_orders={"Segment": ordered_labels},
            hover_data=["Customer ID", "Frequency"],
            title="Recency vs monetary by segment",
            labels={"Recency": "Days since last purchase", "Monetary": "Total Spend (£)"},
            opacity=0.65,
        )
        fig_scatter.update_xaxes(autorange="reversed")
        fig_scatter.update_yaxes(
            type="log",
            tickvals=[1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
            ticktext=["£1", "£10", "£100", "£1,000", "£10,000", "£100,000", "£1,000,000"],
        )
        fig_scatter.update_traces(marker_size=4)

    finalise_fig(fig_scatter)
    st.plotly_chart(fig_scatter, width="stretch")

    # ── Radar + Box: side by side ─────────────────────────────────────────
    st.space("small")
    col_a, col_b = st.columns(2)

    with col_a:
        radar_df = (
            rfm_display.groupby("Segment", observed=False)[["Recency", "Frequency", "Monetary"]]
            .mean()
            .dropna()
        )

        radar_norm = radar_df.copy()
        rec_max = radar_norm["Recency"].max()
        radar_norm["Recency"] = 1 - (radar_norm["Recency"] / rec_max) if rec_max else radar_norm["Recency"]

        for col in ["Frequency", "Monetary"]:
            col_max = radar_norm[col].max()
            radar_norm[col] = radar_norm[col] / col_max if col_max else radar_norm[col]

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
                line=dict(color=seg_colors.get(str(segment), CHART_COLORWAY[0]), width=2),
                opacity=0.8,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=NEUTRAL_RADAR_GRID),
                angularaxis=dict(gridcolor=NEUTRAL_RADAR_GRID),
                bgcolor="rgba(253,253,248,0.55)",
            ),
            dragmode=False,
            title="Normalised RFM profile per segment"
        )
        finalise_fig(fig_radar)
        st.plotly_chart(fig_radar, width="stretch")

    with col_b:
        fig_box = px.box(
            rfm_display,
            x="Segment",
            y="Monetary",
            color="Segment",
            color_discrete_map=seg_colors,
            category_orders={"Segment": ordered_labels},
            title="Spend distribution by segment",
            labels={"Monetary": "Total Spend (£)"},
        )
        fig_box.update_layout(showlegend=False, dragmode=False)
        fig_box.update_yaxes(
            type="log",
            tickvals=[1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
            ticktext=["£1", "£10", "£100", "£1,000", "£10,000", "£100,000", "£1,000,000"],
        )
        finalise_fig(fig_box)
        st.plotly_chart(fig_box, width="stretch")

    # ── Download ─────────
    st.space("small")
    section("Export", eyebrow="Download results")
    csv = rfm_display[
        ["Customer ID", "Recency", "Frequency", "Monetary", "Segment"]
    ].to_csv(index=False)

    st.download_button(
        label="Download segmented customers (CSV)",
        data=csv,
        file_name="rfm_segments.csv",
        mime="text/csv",
        icon=":material/download:",
    )
