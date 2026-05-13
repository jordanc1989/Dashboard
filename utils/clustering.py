import time
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.stats import boxcox

from utils.transforms import transform_rfm

MAX_SEGMENTS = 6

_SEGMENT_PALETTE = [
    "#B85F3D",  # warm red-brown — best
    "#2E7D68",  # teal
    "#7A52B3",  # purple
    "#2C78B7",  # blue
    "#B6861E",  # gold
    "#433D37",  # dark grey-brown — worst
]

ALGORITHM_LABELS = {
    "kmeans": "K-means",
    "gmm": "Gaussian mixture",
    "agglomerative": "Agglomerative",
    "hdbscan": "HDBSCAN",
}


def _rfm_tier(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "mid"
    return "low"


def assign_segment_labels(rfm):
    """Rank clusters best→worst and return (segment_series, color_map, ordered_labels).

    Descriptors are derived from each cluster's normalised R/F/M profile so they
    convey meaning rather than arbitrary letters.
    """
    stats = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    rng = (stats.max() - stats.min()).replace(0, 1)
    r_norm = (stats["Recency"] - stats["Recency"].min()) / rng["Recency"]
    f_norm = (stats["Frequency"] - stats["Frequency"].min()) / rng["Frequency"]
    m_norm = (stats["Monetary"] - stats["Monetary"].min()) / rng["Monetary"]
    score = (1 - r_norm) + f_norm + m_norm
    ranked = score.sort_values(ascending=False).index

    r_inv = 1 - r_norm  # 1 = very recent, 0 = dormant
    _R = {"high": "Recent", "mid": "Active", "low": "Dormant"}
    _M = {"high": "High spend", "mid": "Mid spend", "low": "Low spend"}
    _F = {"high": "Frequent", "mid": "Regular", "low": "Infrequent"}

    def part(cid, mapping, norm):
        return mapping[_rfm_tier(float(norm[cid]))]

    # 2-part labels: Monetary · Recency
    label_list = [f"{part(c, _M, m_norm)}, {part(c, _R, r_inv)}" for c in ranked]

    # Add frequency if any 2-part labels collide
    if len(set(label_list)) < len(label_list):
        label_list = [
            f"{part(c, _M, m_norm)}, {part(c, _R, r_inv)}, {part(c, _F, f_norm)}"
            for c in ranked
        ]

    # Last resort: append rank index
    if len(set(label_list)) < len(label_list):
        label_list = [f"{lbl} ({i + 1})" for i, lbl in enumerate(label_list)]

    label_map = dict(zip(ranked, label_list))
    color_map = {lbl: _SEGMENT_PALETTE[i % len(_SEGMENT_PALETTE)] for i, lbl in enumerate(label_list)}
    return rfm["Cluster"].map(label_map), color_map, label_list


def _winsorise(rfm, winsorise_pct):
    rfm = rfm.copy()
    if winsorise_pct < 100:
        q = winsorise_pct / 100
        for col in ["Recency", "Frequency", "Monetary"]:
            rfm[col] = rfm[col].clip(
                lower=rfm[col].quantile(1 - q), upper=rfm[col].quantile(q)
            )
    return rfm


def _fit_algorithm(algorithm, X, n_clusters, min_cluster_size):
    """Fit a clustering algorithm.

    Returns ``(labels, confidence)`` where confidence is a per-customer score
    in [0, 1] for probabilistic models (GMM max posterior, HDBSCAN membership
    probability) and ``None`` for K-means / Agglomerative.
    """
    if algorithm == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        return km.fit_predict(X), None
    if algorithm == "gmm":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=10,
            n_init=5,
        )
        labels = gmm.fit_predict(X)
        confidence = gmm.predict_proba(X).max(axis=1)
        return labels, confidence
    if algorithm == "agglomerative":
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return agg.fit_predict(X), None
    if algorithm == "hdbscan":
        hdb = HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=5, cluster_selection_method="eom", copy=True)
        labels = hdb.fit_predict(X)
        return labels, hdb.probabilities_
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _safe_silhouette(X, labels):
    """Silhouette ignoring HDBSCAN noise (-1); returns nan when <2 valid clusters."""
    mask = labels != -1
    valid = labels[mask]
    if len(np.unique(valid)) < 2 or mask.sum() <= len(np.unique(valid)):
        return float("nan")
    return float(silhouette_score(X[mask], valid))


@st.cache_data(max_entries=16)
def run_clustering(rfm_raw, n_clusters, winsorise_pct=99, algorithm="kmeans", min_cluster_size=30):
    rfm = _winsorise(rfm_raw, winsorise_pct)
    rfm, X = transform_rfm(rfm)
    labels, _ = _fit_algorithm(algorithm, X, n_clusters, min_cluster_size)
    rfm["Cluster"] = labels
    sil = _safe_silhouette(X, rfm["Cluster"].to_numpy())
    return rfm, sil


@st.cache_data(max_entries=8)
def run_all_algorithms(rfm_raw, n_clusters, winsorise_pct=99, min_cluster_size=30):
    """Fit K-means / GMM / Agglomerative / HDBSCAN on the same scaled RFM matrix.

    Returns a dict keyed by algorithm code with fitted labels, the shared scaled
    matrix `X`, fit time, cluster count, outlier count, and quality metrics.
    The comparison tab consumes this directly so no algorithm is re-fit.
    """
    rfm = _winsorise(rfm_raw, winsorise_pct)
    _, X = transform_rfm(rfm)

    results = {}
    for code in ("kmeans", "gmm", "agglomerative", "hdbscan"):
        t0 = time.perf_counter()
        labels, confidence = _fit_algorithm(code, X, n_clusters, min_cluster_size)
        fit_seconds = time.perf_counter() - t0

        mask = labels != -1
        valid = labels[mask]
        unique_clusters = np.unique(valid)
        n_outliers = int((labels == -1).sum())

        if len(unique_clusters) >= 2 and mask.sum() > len(unique_clusters):
            db = float(davies_bouldin_score(X[mask], valid))
            ch = float(calinski_harabasz_score(X[mask], valid))
            sil = float(silhouette_score(X[mask], valid))
        else:
            db = ch = sil = float("nan")

        results[code] = {
            "labels": labels,
            "confidence": confidence,
            "rfm": rfm.assign(Cluster=labels).reset_index(drop=True),
            "fit_seconds": fit_seconds,
            "n_clusters": int(len(unique_clusters)),
            "n_outliers": n_outliers,
            "metrics": {
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
            },
        }

    return {"X": X, "rfm_winsorised": rfm.reset_index(drop=True), "by_algo": results}


@st.cache_data(max_entries=8)
def pca_project(X):
    """Project the scaled RFM matrix to 2D via PCA — shared axes for all algorithms."""
    pca = PCA(n_components=2, random_state=10)
    coords = pca.fit_transform(X)
    df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    variance = tuple(float(v) for v in pca.explained_variance_ratio_)
    return df, variance


@st.cache_data(max_entries=16)
def elbow_data(rfm_raw, winsorise_pct=99, max_segments=6):
    rfm = _winsorise(rfm_raw, winsorise_pct)
    _, X = transform_rfm(rfm)

    max_k = min(MAX_SEGMENTS, len(rfm) - 1)
    if max_k < 2:
        return [], [], []

    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    return list(k_range), inertias, silhouettes
