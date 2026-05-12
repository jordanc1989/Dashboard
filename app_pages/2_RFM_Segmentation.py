import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, QhullError
from utils import (
    CHART_COLORWAY,
    load_data,
    apply_sidebar_filters,
    build_rfm,
    run_clustering,
    run_all_algorithms,
    pca_project,
    elbow_data,
    assign_segment_labels,
    ALGORITHM_LABELS,
    MAX_SEGMENTS,
    render_page_header,
    section,
    NEUTRAL_GRID,
    NEUTRAL_RADAR_GRID,
    finalise_fig,
)


def _hex_to_rgba(hex_c, alpha):
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(
    page_title="RFM segmentation",
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

    tab_baseline, tab_compare = st.tabs(["Baseline (K-means)", "Compare algorithms"])

    with tab_baseline:
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
        st.caption(
            "Spend axis is log-scaled so the gap between £10 and £100 spenders "
            "is visible alongside the gap between £1,000 and £10,000 spenders."
        )

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

    with tab_compare:
        st.space("small")
        st.markdown(
            "Fit four clustering algorithms on the same scaled RFM matrix, then compare "
            "their quality metrics, segment shapes, and customer assignments side by side. "
            "K-means / GMM / Agglomerative share the `k` slider below. HDBSCAN is "
            "density-based and discovers its own cluster count from `min_cluster_size`."
        )

        with st.container(border=True):
            cmp_cols = st.columns([2, 2, 2])
            with cmp_cols[0]:
                cmp_k = st.select_slider(
                    "Number of segments (k)",
                    options=list(range(2, max_k + 1)),
                    value=default_k,
                    key="cmp_k",
                    help="Applies to K-means, GMM and Agglomerative.",
                )
            with cmp_cols[1]:
                cmp_min_size = st.slider(
                    "HDBSCAN min cluster size",
                    min_value=10,
                    max_value=max(20, min(300, len(rfm_raw) // 5)),
                    value=min(30, max(10, len(rfm_raw) // 20)),
                    step=5,
                    key="cmp_min_size",
                    help="Smaller values find more (and smaller) clusters; larger values "
                    "merge everything into a few broad groups.",
                )
            with cmp_cols[2]:
                cmp_winsorise = st.slider(
                    "Winsorise percentile",
                    min_value=95,
                    max_value=100,
                    value=99,
                    step=1,
                    key="cmp_winsorise",
                    help="Outlier clipping applied identically to all four algorithms.",
                )

        results = run_all_algorithms(
            rfm_raw,
            n_clusters=cmp_k,
            winsorise_pct=cmp_winsorise,
            min_cluster_size=cmp_min_size,
        )
        X_shared = results["X"]
        by_algo = results["by_algo"]

        # Build labelled per-algorithm frames (handle HDBSCAN noise as "Outliers")
        def _label_algo(algo_code):
            entry = by_algo[algo_code]
            labels = entry["labels"]
            rfm_winsor = entry["rfm"]

            noise_mask = labels == -1
            if noise_mask.any() and (~noise_mask).sum() >= 2 and len(np.unique(labels[~noise_mask])) >= 1:
                clean = rfm_winsor.loc[~noise_mask].copy()
                if clean["Cluster"].nunique() >= 1:
                    seg_series, color_map, ordered = assign_segment_labels(clean)
                    seg_full = pd.Series(index=rfm_winsor.index, dtype=object)
                    seg_full.loc[~noise_mask] = seg_series.values
                    seg_full.loc[noise_mask] = "Outliers"
                    color_map = {**color_map, "Outliers": NEUTRAL_GRID}
                    ordered = list(ordered) + ["Outliers"]
                else:
                    seg_full = pd.Series(["Outliers"] * len(rfm_winsor), index=rfm_winsor.index)
                    color_map = {"Outliers": NEUTRAL_GRID}
                    ordered = ["Outliers"]
            elif noise_mask.all() or rfm_winsor["Cluster"].nunique() < 2:
                seg_full = pd.Series(
                    ["Outliers" if n else "Cluster 0" for n in noise_mask],
                    index=rfm_winsor.index,
                )
                color_map = {"Cluster 0": CHART_COLORWAY[0], "Outliers": NEUTRAL_GRID}
                ordered = ["Cluster 0", "Outliers"]
            else:
                seg_series, color_map, ordered = assign_segment_labels(rfm_winsor)
                seg_full = seg_series

            return seg_full, color_map, ordered

        labelled = {code: _label_algo(code) for code in by_algo}

        # ── Viz 1: Scorecard table ───────────────────────────────────────
        st.space("small")
        section("Model scorecard", eyebrow="Quality metrics")

        scorecard_rows = []
        for code, entry in by_algo.items():
            labels = entry["labels"]
            valid = labels[labels != -1]
            total = len(labels)
            if len(valid):
                sizes = pd.Series(valid).value_counts(normalize=True)
                smallest_pct = float(sizes.min() * 100)
            else:
                smallest_pct = float("nan")
            m = entry["metrics"]
            scorecard_rows.append({
                "Model": ALGORITHM_LABELS[code],
                "Clusters": entry["n_clusters"],
                "Silhouette": m["silhouette"],
                "Davies-Bouldin": m["davies_bouldin"],
                "Calinski-Harabasz": m["calinski_harabasz"],
                "Outlier %": (entry["n_outliers"] / total * 100) if total else 0.0,
                "Smallest segment %": smallest_pct,
                "Fit time (s)": entry["fit_seconds"],
            })
        scorecard = pd.DataFrame(scorecard_rows)

        styled = (
            scorecard.style
            .background_gradient(subset=["Silhouette", "Calinski-Harabasz"], cmap="Greens")
            .background_gradient(subset=["Davies-Bouldin"], cmap="Reds_r")
            .format({
                "Silhouette": "{:.3f}",
                "Davies-Bouldin": "{:.3f}",
                "Calinski-Harabasz": "{:,.0f}",
                "Outlier %": "{:.1f}%",
                "Smallest segment %": "{:.1f}%",
                "Fit time (s)": "{:.2f}",
            }, na_rep="—")
        )
        st.dataframe(styled, width="stretch", hide_index=True)

        # ── Viz 2: Metric bar charts ─────────────────────────────────────
        st.space("small")
        section("Metric comparison", eyebrow="By model")

        def _metric_bar(values, title, color):
            fig = go.Figure(go.Bar(
                x=[ALGORITHM_LABELS[c] for c in by_algo.keys()],
                y=values,
                marker_color=color,
            ))
            fig.update_layout(title=title, yaxis_title=title)
            finalise_fig(fig)
            return fig

        bar_cols_a = st.columns(2)
        with bar_cols_a[0]:
            st.plotly_chart(
                _metric_bar(
                    [by_algo[c]["metrics"]["silhouette"] for c in by_algo],
                    "Silhouette (higher = better)",
                    CHART_COLORWAY[1],
                ),
                width="stretch",
            )
        with bar_cols_a[1]:
            st.plotly_chart(
                _metric_bar(
                    [by_algo[c]["metrics"]["davies_bouldin"] for c in by_algo],
                    "Davies-Bouldin (lower = better)",
                    CHART_COLORWAY[0],
                ),
                width="stretch",
            )

        bar_cols_b = st.columns(2)
        with bar_cols_b[0]:
            outlier_pct = [
                (by_algo[c]["n_outliers"] / len(by_algo[c]["labels"]) * 100)
                for c in by_algo
            ]
            st.plotly_chart(
                _metric_bar(outlier_pct, "Outlier %", CHART_COLORWAY[4]),
                width="stretch",
            )
        with bar_cols_b[1]:
            st.plotly_chart(
                _metric_bar(
                    [by_algo[c]["fit_seconds"] for c in by_algo],
                    "Fit time (s)",
                    CHART_COLORWAY[3],
                ),
                width="stretch",
            )

        # ── Viz 3: Side-by-side PCA scatter ─────────────────────────────
        st.space("small")
        section("PCA projection", eyebrow="Shared 2D map")
        pca_df, variance = pca_project(X_shared)
        pc1_pct, pc2_pct = variance[0] * 100, variance[1] * 100
        st.caption(
            f"PC1 ({pc1_pct:.1f}% var) · PC2 ({pc2_pct:.1f}% var) · "
            f"{pc1_pct + pc2_pct:.1f}% total — all four scatters share the same "
            "projection so cluster shapes are directly comparable."
        )

        pca_cols_a = st.columns(2)
        pca_cols_b = st.columns(2)
        slots = [pca_cols_a[0], pca_cols_a[1], pca_cols_b[0], pca_cols_b[1]]

        for slot, code in zip(slots, by_algo.keys()):
            seg_full, color_map, ordered = labelled[code]
            plot_df = pca_df.copy()
            plot_df["Segment"] = seg_full.values
            fig_pca = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="Segment",
                color_discrete_map=color_map,
                category_orders={"Segment": ordered},
                title=ALGORITHM_LABELS[code],
                opacity=0.65,
            )
            fig_pca.update_traces(marker_size=4)
            fig_pca.update_layout(
                xaxis_title=f"PC1 ({pc1_pct:.1f}%)",
                yaxis_title=f"PC2 ({pc2_pct:.1f}%)",
                legend=dict(font=dict(size=10)),
            )

            # Convex-hull segment dividers — outline + light fill per segment.
            # Skip the "Outliers" bucket (HDBSCAN noise has no meaningful boundary).
            for seg in ordered:
                if seg == "Outliers":
                    continue
                pts = plot_df.loc[plot_df["Segment"] == seg, ["PC1", "PC2"]].to_numpy()
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                except QhullError:
                    continue
                ring = pts[hull.vertices]
                ring = np.vstack([ring, ring[0]])
                seg_color = color_map.get(seg, CHART_COLORWAY[0])
                fig_pca.add_trace(go.Scatter(
                    x=ring[:, 0],
                    y=ring[:, 1],
                    mode="lines",
                    line=dict(color=seg_color, width=1.5),
                    fill="toself",
                    fillcolor=_hex_to_rgba(seg_color, 0.08),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            finalise_fig(fig_pca)
            with slot:
                st.plotly_chart(fig_pca, width="stretch")

        # ── Viz 4: Segment-size grouped bar ─────────────────────────────
        st.space("small")
        section("Segment balance", eyebrow="Customer share")

        size_rows = []
        for code in by_algo:
            seg_full, _, _ = labelled[code]
            counts = seg_full.value_counts(normalize=True) * 100
            for seg, pct in counts.items():
                size_rows.append({
                    "Model": ALGORITHM_LABELS[code],
                    "Segment": seg,
                    "Share %": float(pct),
                })
        size_df = pd.DataFrame(size_rows)

        fig_size = px.bar(
            size_df,
            x="Model",
            y="Share %",
            color="Segment",
            barmode="group",
            title="Customer share by segment within each model",
        )
        fig_size.update_layout(yaxis_title="Customers (%)")
        finalise_fig(fig_size)
        st.plotly_chart(fig_size, width="stretch")

        # ── Viz 5: RFM profile heatmap ──────────────────────────────────
        st.space("small")
        section("RFM profile heatmap", eyebrow="Mean R/F/M per (model, segment)")
        st.caption(
            "Values are z-scored per RFM dimension across all rows so colour is comparable "
            "across columns. Recency is *not* inverted here — darker = more days since last "
            "purchase (less recent)."
        )

        profile_rows = []
        for code in by_algo:
            seg_full, _, ordered = labelled[code]
            rfm_winsor = by_algo[code]["rfm"].copy()
            rfm_winsor["Segment"] = seg_full.values
            for seg in ordered:
                mask = rfm_winsor["Segment"] == seg
                if not mask.any():
                    continue
                sub = rfm_winsor.loc[mask, ["Recency", "Frequency", "Monetary"]].mean()
                profile_rows.append({
                    "Row": f"{ALGORITHM_LABELS[code]}: {seg}",
                    "Recency": float(sub["Recency"]),
                    "Frequency": float(sub["Frequency"]),
                    "Monetary": float(sub["Monetary"]),
                })

        profile_df = pd.DataFrame(profile_rows).set_index("Row")
        # z-score per column so colours are comparable
        profile_z = (profile_df - profile_df.mean()) / profile_df.std(ddof=0).replace(0, 1)

        fig_heat = go.Figure(go.Heatmap(
            z=profile_z.values,
            x=profile_z.columns.tolist(),
            y=profile_z.index.tolist(),
            colorscale=[
                [0.0, "#2E7D68"],
                [0.5, "#fdfdf8"],
                [1.0, "#B85F3D"],
            ],
            zmid=0,
            colorbar=dict(title="z-score"),
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="RFM profile per (model, segment)",
            height=max(360, 28 * len(profile_df) + 80),
            yaxis=dict(autorange="reversed"),
        )
        finalise_fig(fig_heat)
        st.plotly_chart(fig_heat, width="stretch")

        # ── Viz 8: Revenue share by segment per model ───────────────────
        st.space("small")
        section("Revenue concentration", eyebrow="Where does the money sit?")
        st.caption(
            "Revenue share = sum of Monetary within each segment ÷ total Monetary for that "
            "model. A balanced model spreads revenue across segments; a useful model often "
            "concentrates it — a small VIP segment carrying a big share of revenue."
        )

        revenue_rows = []
        for code in by_algo:
            seg_full, _, ordered = labelled[code]
            rfm_winsor = by_algo[code]["rfm"].copy()
            rfm_winsor["Segment"] = seg_full.values
            totals = rfm_winsor.groupby("Segment", observed=False)["Monetary"].sum()
            total_rev = totals.sum()
            if total_rev <= 0:
                continue
            for seg in ordered:
                if seg not in totals.index:
                    continue
                revenue_rows.append({
                    "Model": ALGORITHM_LABELS[code],
                    "Segment": seg,
                    "Revenue share %": float(totals[seg] / total_rev * 100),
                })
        revenue_df = pd.DataFrame(revenue_rows)

        fig_rev = px.bar(
            revenue_df,
            x="Model",
            y="Revenue share %",
            color="Segment",
            barmode="stack",
            title="Revenue share by segment within each model",
        )
        fig_rev.update_layout(yaxis_title="Revenue (%)")
        finalise_fig(fig_rev)
        st.plotly_chart(fig_rev, width="stretch")

        # ── Viz 9: Confidence distribution ──────────────────────────────
        st.space("small")
        section("Assignment confidence", eyebrow="GMM & HDBSCAN")
        st.caption(
            "Probabilistic models report how sure they are about each customer's "
            "placement. GMM: max posterior probability. HDBSCAN: cluster membership "
            "probability (noise points = 0). Big peaks near 1 mean confident assignments; "
            "mass near 0.5 signals customers the model is guessing on."
        )

        conf_models = [c for c in by_algo if by_algo[c]["confidence"] is not None]
        if not conf_models:
            st.info("No probabilistic models available in the current selection.")
        else:
            conf_cols = st.columns(len(conf_models))
            for slot, code in zip(conf_cols, conf_models):
                conf_values = np.asarray(by_algo[code]["confidence"], dtype=float)
                fig_conf = go.Figure(go.Histogram(
                    x=conf_values,
                    nbinsx=30,
                    marker_color=CHART_COLORWAY[1] if code == "gmm" else CHART_COLORWAY[2],
                ))
                p_low = float((conf_values < 0.6).mean() * 100)
                fig_conf.update_layout(
                    title=f"{ALGORITHM_LABELS[code]} — {p_low:.1f}% below 0.6",
                    xaxis_title="Confidence",
                    yaxis_title="Customers",
                    xaxis=dict(range=[0, 1]),
                    bargap=0.02,
                )
                finalise_fig(fig_conf)
                with slot:
                    st.plotly_chart(fig_conf, width="stretch")

        # ── Viz 10: Business-label diff table ───────────────────────────
        st.space("small")
        section("Business label diff", eyebrow="Would decisions change?")
        st.caption(
            "Customer counts per business-label persona across models. Blank cells mean "
            "that model did not produce that persona. Compare rows to see if the "
            "algorithms agree on size of each commercially meaningful group."
        )

        diff_rows = []
        for code in by_algo:
            seg_full, _, _ = labelled[code]
            counts = seg_full.value_counts()
            for seg, n in counts.items():
                diff_rows.append({
                    "Segment": seg,
                    "Model": ALGORITHM_LABELS[code],
                    "Customers": int(n),
                })
        diff_df = pd.DataFrame(diff_rows)
        diff_pivot = diff_df.pivot(index="Segment", columns="Model", values="Customers")
        diff_pivot = diff_pivot.reindex(columns=[ALGORITHM_LABELS[c] for c in by_algo])

        # Sort: persona-style labels first (best→worst), Outliers last
        def _sort_key(label):
            if label == "Outliers":
                return (2, label)
            if "spend" in label.lower():
                return (0, label)
            return (1, label)

        diff_pivot = diff_pivot.loc[sorted(diff_pivot.index, key=_sort_key)]

        styled_diff = (
            diff_pivot.style
            .background_gradient(cmap="Greens", axis=None)
            .format("{:,.0f}", na_rep="—")
        )
        st.dataframe(styled_diff, width="stretch")
