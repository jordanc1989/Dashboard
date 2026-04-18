import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from utils import (
    CHART_COLORWAY,
    COLOR_SCALE_EXPECTED_PURCHASES,
    NEUTRAL_GRID,
    apply_sidebar_filters,
    build_churn_dataset,
    load_data,
    render_dataset_subtitle,
    finalize_fig,
)

st.set_page_config(
    page_title="Churn Prediction · Customer Analytics",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

df = load_data()
df = apply_sidebar_filters(df)

render_dataset_subtitle(df)
st.markdown(
    "Non-contractual retail has no explicit churn label, so we fabricate one "
    "from a **time split**. Customers active on or before a cutoff date are "
    "labelled *churned* if they make no further purchase in the subsequent "
    "window. A **Random Forest** classifier is then trained on engineered "
    "RFM-style features to predict this outcome."
)

with st.expander("How churn is defined here", expanded=False):
    st.markdown(
        """
The Online Retail dataset has no subscription or cancellation signal. The
standard workaround for non-contractual retail is a **rolling-window definition**:

1. Pick a churn window (default 90 days).
2. Let `cutoff = max_invoice_date - churn window`.
3. For every registered customer active on or before `cutoff`, compute features
   **using only transactions up to `cutoff`** (no look-ahead). 
4. Label the customer **churned = 1** if they made no purchase in the window, else 0.

This gives a real, held-out ground truth from the data itself. The model is
then trained and evaluated on an 80 / 20 stratified split of this labelled set,
and out-of-fold probabilities are generated for every customer via 5-fold
cross-validation for the at-risk table below.
        """
    )

# ── Controls ─────────────────────────────────────────────────────────────────
span_days = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
max_window = max(30, min(365, span_days // 3))

c1, c2, c3 = st.columns(3)
with c1:
    window_days = st.slider(
        "Churn window (days)",
        min_value=30,
        max_value=max_window,
        value=min(90, max_window),
        step=15,
        help=(
            "No purchase within this window after the cutoff = churned. "
            "Shorter windows → more customers labelled churned and a noisier "
            "signal; longer windows → fewer churners but more stable."
        ),
    )
with c2:
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Probability above which a customer is classified as churned. "
            "Lower thresholds catch more churners (higher recall) at the "
            "cost of more false positives."
        ),
    )
with c3:
    balance_classes = st.toggle(
        "Balance classes",
        value=True,
        help=(
            "When churners are the minority class, this prevents the model from "
            "ignoring them. Boosts recall at the cost of some precision: usually "
            "the right trade-off for retention. Watch the confusion matrix change."
        ),
    )

with st.expander("Advanced model settings", expanded=False):
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        n_estimators = st.slider(
            "Number of trees",
            min_value=50,
            max_value=300,
            value=200,
            step=50,
            help="More trees gives smoother probabilities but slower fit.",
        )
    with adv2:
        max_depth = st.slider(
            "Max tree depth",
            min_value=1,
            max_value=15,
            value=10,
            help=(
                "Limits how deep each tree grows. Higher values allow more complex splits "
                "(risk overfitting on small samples). 5-15 is often a good range for retail churn."
            ),
        )
    with adv3:
        min_samples_split = st.slider(
            "Min samples to split",
            min_value=2,
            max_value=10,
            value=5,
            help=(
                "A node must have at least this many samples before it can be split further. "
                "Higher = more regularisation."
            ),
        )

# ── Build dataset ────────────────────────────────────────────────────────────
features, meta = build_churn_dataset(df, churn_window_days=int(window_days))

if len(features) < 100 or features["churned"].nunique() < 2:
    st.warning(
        "Not enough labelled customers in this selection to train a classifier. "
        "Widen the date range or 'All' countries in the sidebar, or adjust the "
        "churn window."
    )
    st.stop()

feature_cols = [
    "recency_days",
    "tenure_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_items_per_order",
    "unique_products",
    "return_rate",
]
X = features[feature_cols].values
y = features["churned"].values


# ── Train / evaluate ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training random forest...")
def fit_and_score(
    X,
    y,
    n_estimators: int,
    balance: bool,
    window_days: int,
    n_rows: int,
    max_depth: int,
    min_samples_split: int,
):
    """Returns (test_metrics_dict, oof_probs, feat_importances, fitted_model)."""
    rf_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=10,
    )
    if balance:
        rf_kwargs["class_weight"] = "balanced"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=10
    )
    model = RandomForestClassifier(**rf_kwargs)
    model.fit(X_tr, y_tr)

    train_proba = model.predict_proba(X_tr)[:, 1]
    train_auc = roc_auc_score(y_tr, train_proba)

    test_proba = model.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, test_proba)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    fold_aucs = cross_val_score(
        RandomForestClassifier(**rf_kwargs),
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    test_metrics = {
        "auc": test_auc,
        "train_auc": train_auc,
        "fold_aucs": fold_aucs,
        "y_test": y_te,
        "proba_test": test_proba,
    }

    oof = cross_val_predict(
        RandomForestClassifier(**rf_kwargs), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    return test_metrics, oof, importances, model


test_metrics, oof_probs, importances, _ = fit_and_score(
    X,
    y,
    int(n_estimators),
    bool(balance_classes),
    int(window_days),
    len(features),
    int(max_depth),
    int(min_samples_split),
)

y_pred_test = (test_metrics["proba_test"] >= threshold).astype(int)
prec = precision_score(test_metrics["y_test"], y_pred_test, zero_division=0)
rec = recall_score(test_metrics["y_test"], y_pred_test, zero_division=0)

features = features.assign(churn_prob=oof_probs)


# ── KPIs ─────────────────────────────────────────────────────────────────────
actual_churn_rate = meta["churn_rate"]
predicted_churn_rate = (features["churn_prob"] >= threshold).mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Actual churn rate",
    f"{actual_churn_rate * 100:.1f}%",
    help=f"Share of {meta['n_customers']:,} customers who did not return in the {meta['window_days']}-day window.",
)
k2.metric(
    "Predicted churn rate",
    f"{predicted_churn_rate * 100:.1f}%",
    delta=f"{(predicted_churn_rate - actual_churn_rate) * 100:+.1f} pp",
    delta_color="off",
    help="Share of customers whose out-of-fold probability exceeds the decision threshold.",
)
k3.metric("AUC-ROC", f"{test_metrics['auc']:.3f}", help="Hold-out area under the ROC curve. 0.5 = random, 1.0 = perfect.")
k4.metric("Precision", f"{prec:.2f}", help="Of customers we flag as churners, how many actually churn.")
k5.metric("Recall", f"{rec:.2f}", help="Of customers who actually churn, how many we catch.")

overfit_gap = test_metrics["train_auc"] - test_metrics["auc"]
fold_std = float(test_metrics["fold_aucs"].std())

st.caption(
    f"Train AUC: **{test_metrics['train_auc']:.3f}**  ·  "
    f"Test AUC: **{test_metrics['auc']:.3f}**  ·  "
    f"Gap: **{overfit_gap:+.3f}**  ·  "
    f"CV fold std: **{fold_std:.3f}**"
)

if overfit_gap > 0.08:
    st.warning(
        f"Warning: Train AUC ({test_metrics['train_auc']:.3f}) is notably higher than "
        f"test AUC ({test_metrics['auc']:.3f}): the model may be overfitting. "
        f"Try reducing *max tree depth* or increasing *min samples to split*."
    )

st.info(
    "**Recall matters more than precision for retention.** Contacting a "
    "loyal customer with an unnecessary offer is cheap, but failing to flag an "
    "at-risk one could mean lost revenue! Lower the **decision threshold** above to "
    "trade precision for recall."
)

st.caption(
    f"Cutoff: **{meta['cutoff'].date()}**  ·  window: **{meta['window_days']} days**  "
    f"·  customers labelled: **{meta['n_customers']:,}**  ·  "
    f"churners: **{int(features['churned'].sum()):,}**"
)

# ── Feature importance + ROC ─────────────────────────────────────────────────
st.subheader("Model Explainability")
left, right = st.columns(2)

with left:
    imp_df = importances.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()
    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (Gini)",
        color_discrete_sequence=[CHART_COLORWAY[0]],
    )
    fig_imp.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Importance",
    )
    finalize_fig(fig_imp)
    st.plotly_chart(fig_imp, width="stretch")

with right:
    fpr, tpr, _ = roc_curve(test_metrics["y_test"], test_metrics["proba_test"])
    prec_curve, rec_curve, _ = precision_recall_curve(
        test_metrics["y_test"], test_metrics["proba_test"]
    )
    pr_auc = auc(rec_curve, prec_curve)

    fig_roc = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "ROC (hold-out 20%)",
            "Precision–Recall (hold-out 20%)",
        ),
        horizontal_spacing=0.1,
    )
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={test_metrics['auc']:.3f})",
            line=dict(color=CHART_COLORWAY[1], width=2.5),
        ),
        row=1,
        col=1,
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(color=NEUTRAL_GRID, width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_roc.add_trace(
        go.Scatter(
            x=rec_curve,
            y=prec_curve,
            mode="lines",
            name=f"PR (AUC={pr_auc:.3f})",
            line=dict(color=CHART_COLORWAY[2], width=2.5),
        ),
        row=1,
        col=2,
    )
    fig_roc.update_xaxes(title_text="False positive rate", range=[0, 1], row=1, col=1)
    fig_roc.update_yaxes(title_text="True positive rate", range=[0, 1], row=1, col=1)
    fig_roc.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    fig_roc.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
    fig_roc.update_layout(
        margin=dict(t=48, b=40, l=12, r=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        height=380,
    )
    finalize_fig(fig_roc)
    st.caption(
        "Left: ROC trades off false positives vs true positives at varying thresholds. "
        "Right: precision vs recall at the same probabilities (better for imbalanced churn rates)."
    )
    st.plotly_chart(fig_roc, width="stretch")

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(test_metrics["y_test"], y_pred_test, labels=[0, 1])
cm_labels = ["Retained", "Churned"]
fig_cm = px.imshow(
    cm,
    x=[f"Pred: {l}" for l in cm_labels],
    y=[f"Actual: {l}" for l in cm_labels],
    color_continuous_scale=COLOR_SCALE_EXPECTED_PURCHASES,
    text_auto=True,
    title=f"Confusion Matrix @ threshold {threshold:.2f} (held-out 20%)",
    aspect="equal",
)
fig_cm.update_coloraxes(showscale=False)
fig_cm.update_xaxes(side="bottom")
finalize_fig(fig_cm)
_pad_l, cm_col, _pad_r = st.columns([3, 4, 3])
with cm_col:
    st.plotly_chart(fig_cm, width="stretch")

# ── At-risk customer table ───────────────────────────────────────────────────
st.subheader("At-Risk Customers")
st.markdown(
    "Out-of-fold predictions for every customer, sortable by churn probability. "
    "Useful for scoping a retention campaign: the top rows are the highest-value "
    "targets to contact first."
)

top_n = st.slider("Show top N by churn probability", min_value=10, max_value=200, value=50, step=10)

display_cols = [
    "Customer ID",
    "churn_prob",
    "churned",
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "orders_per_month",
    "unique_products",
    "n_returns",
    "return_rate",
]
table = (
    features[display_cols]
    .sort_values("churn_prob", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
    .rename(
        columns={
            "churn_prob": "Churn P",
            "churned": "Actual",
            "recency_days": "Recency (days)",
            "frequency": "Orders",
            "monetary": "Spend (£)",
            "avg_order_value": "AOV (£)",
            "tenure_days": "Tenure (days)",
            "orders_per_month": "Orders / month",
            "unique_products": "Unique Products",
            "n_returns": "Returns",
            "return_rate": "Return Rate",
        }
    )
)
table["Actual"] = table["Actual"].map({0: "Retained", 1: "Churned"})

st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "Churn P": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "Spend (£)": st.column_config.NumberColumn(format="£ %,.0f"),
        "AOV (£)": st.column_config.NumberColumn(format="£ %.2f"),
        "Orders / month": st.column_config.NumberColumn(format="%.2f"),
        "Recency (days)": st.column_config.NumberColumn(format="%,d"),
        "Tenure (days)": st.column_config.NumberColumn(format="%,d"),
        "Orders": st.column_config.NumberColumn(format="%,d"),
        "Unique Products": st.column_config.NumberColumn(format="%,d"),
        "Returns": st.column_config.NumberColumn(format="%,d"),
        "Return Rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    },
)

# ── Download ─────────────────────────────────────────────────────────────────
export = features[["Customer ID", "churn_prob", "churned"] + feature_cols].rename(
    columns={"churn_prob": "churn_probability", "churned": "actual_churn"}
)
csv = export.to_csv(index=False)
st.download_button(
    label="Download all churn scores (CSV)",
    data=csv,
    file_name="churn_predictions.csv",
    mime="text/csv",
)

st.caption(
    "Churn labels come from an internal time split, so accuracy figures reflect "
    "how well the model predicts dormancy in the final window of the dataset, "
    "not an unseen future. For production use you would retrain on the full "
    "history and score current customers using today as the cutoff."
)
