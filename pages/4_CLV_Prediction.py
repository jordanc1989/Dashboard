import hashlib
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
from pymc_marketing.clv.utils import customer_lifetime_value, rfm_train_test_split
from utils import load_data, apply_sidebar_filters, build_clv_summary

WEEKS_PER_MONTH = 4.345  # 365.25 / 12 / 7


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


st.set_page_config(
    page_title="CLV Prediction · Customer Analytics",
    page_icon="💰",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

st.header("Customer Lifetime Value Prediction")
st.markdown(
    "Two probabilistic models are chained to predict future value per customer: "
    "**BG/NBD** (Beta-Geometric / Negative Binomial Distribution) models *when* customers "
    "will purchase again, and **Gamma-Gamma** models *how much* they'll spend. "
    "Together they produce an individual CLV forecast over any horizon you choose."
)

with st.expander("How the models work"):
    st.markdown("""
**BG/NBD model** assumes each customer has an unobserved purchase rate (Poisson) and dropout
probability (Geometric). The population-level distributions of these rates are modelled as
Gamma and Beta respectively — hence the name. It takes three inputs per customer:

- **Frequency** — number of repeat purchases (total orders minus the first)
- **Recency** — weeks since their last purchase
- **T** — total weeks since their first purchase (customer age)

It outputs **expected future transactions** over a chosen time window and **probability the
customer is still active** ("alive").

**Gamma-Gamma model** takes customers with at least one repeat purchase and models the
distribution of average spend, assuming spend is independent of purchase frequency.
It outputs a corrected estimate of **expected average order value**.""")

    st.latex(r"""
\mathrm{CLV} = \sum_{t=1}^{T}
\frac{\mathbb{E}[\text{purchases in period } t] \times \mathbb{E}[\text{AOV}]}
{(1 + \text{discount rate})^t}
""")

    st.markdown("""
Both models are fitted here using **MAP estimation** (maximum a posteriori), which is
equivalent to maximum likelihood with the Bayesian prior acting as a regulariser.
Full MCMC sampling would additionally quantify uncertainty around each prediction.
    """)

df_customers = df[~df["is_guest"]]

if df_customers["CustomerID"].nunique() < 50:
    st.warning("Not enough registered customers in this selection. Try 'All' countries or widen the date range.")
    st.stop()

summary = build_clv_summary(df_customers)

if len(summary) < 20:
    st.warning("Too few repeat customers to fit the models reliably. Widen the filters.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    horizon_months = st.slider("Prediction horizon (months)", min_value=1, max_value=12, value=3)
with col_ctrl2:
    annual_rate = st.slider(
        "Annual discount rate (%)",
        min_value=0.0, max_value=15.0, value=3.5, step=0.5,
        help="Cost of capital used to discount future cash flows to present value. 3–5% is typical for stable retailers."
    ) / 100
    monthly_discount = (1 + annual_rate) ** (1 / 12) - 1

horizon_weeks = horizon_months * WEEKS_PER_MONTH

# ── Fit models ─────────────────────────────────────────────────────────────────
bg_data = summary[["customer_id", "frequency", "recency", "T"]].copy()
gg_data = summary[["customer_id", "frequency", "monetary_value"]].copy()


@st.cache_resource
def fit_bgnbd(data_hash: str, _data: pd.DataFrame):
    bgm = BetaGeoModel(data=_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bgm.fit(method="map")
    return bgm


@st.cache_resource
def fit_gg(data_hash: str, _data: pd.DataFrame):
    ggm = GammaGammaModel(data=_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ggm.fit(method="map")
    return ggm


with st.spinner("Fitting BG/NBD and Gamma-Gamma models (MAP)…"):
    bgm = fit_bgnbd(_df_hash(bg_data), bg_data)
    ggm = fit_gg(_df_hash(gg_data), gg_data)

# ── Gamma-Gamma independence assumption check ──────────────────────────────────
fm_corr = summary[["frequency", "monetary_value"]].corr().iloc[0, 1]
if abs(fm_corr) > 0.3:
    st.warning(
        f"⚠️ Frequency–monetary correlation is **{fm_corr:+.2f}**. "
        "The Gamma-Gamma model assumes these are approximately independent "
        "(|r| < 0.3). Spend predictions may be biased for this segment."
    )
else:
    st.caption(
        f"✓ Gamma-Gamma independence assumption holds "
        f"(frequency–monetary correlation = {fm_corr:+.2f}, |r| < 0.3)."
    )

# ── Predictions ────────────────────────────────────────────────────────────────
predicted_purchases = (
    bgm.expected_purchases(data=bg_data, future_t=horizon_weeks)
    .mean(("chain", "draw"))
    .to_series()
)
prob_alive = (
    bgm.expected_probability_alive(data=bg_data)
    .mean(("chain", "draw"))
    .to_series()
)
expected_aov = (
    ggm.expected_customer_spend(data=gg_data)
    .mean(("chain", "draw"))
    .to_series()
)

# customer_lifetime_value requires a future_spend column; future_t is in months
clv_data = bg_data.copy()
clv_data["future_spend"] = expected_aov.values

clv = (
    customer_lifetime_value(
        bgm,
        clv_data,
        future_t=horizon_months,
        discount_rate=monthly_discount,
        time_unit="W",
    )
    .mean(("chain", "draw"))
    .to_series()
    .clip(lower=0)
)

summary["predicted_purchases"] = predicted_purchases.values
summary["prob_alive"]           = prob_alive.values
summary["expected_aov"]         = expected_aov.values
summary["clv"]                  = clv.values

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers modelled", f"{len(summary):,}")
k2.metric(f"Expected revenue (next {horizon_months}m)", f"£{summary['clv'].sum():,.0f}")
k3.metric("Median customer CLV", f"£{summary['clv'].median():,.2f}")
k4.metric("Median P(still active)", f"{summary['prob_alive'].median()*100:.1f}%")

st.divider()

# ── Out-of-sample validation ───────────────────────────────────────────────────
@st.cache_data
def run_holdout_validation(_df: pd.DataFrame) -> tuple | None:
    invoices = (
        _df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )

    obs_start = _df["InvoiceDate"].min()
    obs_end   = _df["InvoiceDate"].max()
    cal_end   = obs_start + pd.Timedelta(days=int((obs_end - obs_start).days * 0.75))
    holdout_weeks = (obs_end - cal_end).days / 7.0

    ch = rfm_train_test_split(
        invoices,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        train_period_end=cal_end,
        test_period_end=obs_end,
        time_unit="W",
    )

    ch = ch[ch["frequency"] > 0].copy()
    if len(ch) < 20:
        return None

    cal_bg = ch[["customer_id", "frequency", "recency", "T"]].copy()
    bgm_cal = BetaGeoModel(data=cal_bg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bgm_cal.fit(method="map")

    ch["predicted"] = (
        bgm_cal.expected_purchases(data=cal_bg, future_t=holdout_weeks)
        .mean(("chain", "draw"))
        .to_series()
        .values
    )
    return ch, holdout_weeks, cal_end


with st.expander("📏 Out-of-sample validation (calibration / holdout)", expanded=False):
    st.markdown(
        "Model performance is evaluated by withholding the last 25% of the observation window, "
        "fitting BG/NBD only on the earlier 75%, and predicting the number of purchases each "
        "customer will make in the holdout. The chart bins customers by their frequency in the "
        "calibration period and plots predicted vs. actual mean purchases in the holdout. "
        "Close alignment with the dashed y=x line indicates the model is well-calibrated."
    )

    validation = run_holdout_validation(df_customers)
    if validation is None:
        st.info("Not enough data to run holdout validation for this selection.")
    else:
        ch, holdout_weeks, cal_end = validation

        binned = (
            ch.groupby(ch["frequency"].clip(upper=7))
            .agg(
                actual=("test_frequency", "mean"),
                predicted=("predicted", "mean"),
                n=("predicted", "size"),
            )
            .reset_index()
        )

        mae  = (ch["predicted"] - ch["test_frequency"]).abs().mean()
        rmse = np.sqrt(((ch["predicted"] - ch["test_frequency"]) ** 2).mean())

        m1, m2, m3 = st.columns(3)
        m1.metric("Holdout window", f"{holdout_weeks:.0f} weeks")
        m2.metric("MAE", f"{mae:.2f} purchases")
        m3.metric("RMSE", f"{rmse:.2f} purchases")

        max_val = max(binned["actual"].max(), binned["predicted"].max()) * 1.1
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect prediction",
            line=dict(color="#9CA3AF", width=1.5, dash="dash"),
        ))
        fig_val.add_trace(go.Scatter(
            x=binned["predicted"], y=binned["actual"],
            mode="markers+lines",
            name="Actual vs predicted",
            marker=dict(size=10, color="#B85F3D"),
            line=dict(color="#B85F3D", width=2),
            text=[f"n={n}" for n in binned["n"]],
            hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}<extra></extra>",
        ))
        fig_val.update_layout(
            title=f"Calibration vs holdout (train to {cal_end.date()}, holdout {holdout_weeks:.0f}w)",
            xaxis_title="Mean predicted purchases",
            yaxis_title="Mean actual purchases",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_val, width="stretch")

# ── Model diagnostics ──────────────────────────────────────────────────────────
st.subheader("Model Diagnostics")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("**Frequency–Recency matrix** — expected purchases in the next period")
    st.markdown(
        "Frequent buyers who purchased recently (bottom-left) are predicted to buy most. "
        "Frequent buyers who haven't purchased in a long time (bottom-right) "
        "are likely churned and the model predicts far fewer future purchases."
    )

    t_val    = float(summary["T"].median())
    max_freq = min(int(summary["frequency"].quantile(0.95)), 40)
    max_wsl  = int(t_val)

    freq_grid = np.arange(0, max_freq + 1)
    wsl_grid  = np.arange(0, max_wsl + 1)

    freq_2d, wsl_2d = np.meshgrid(freq_grid, wsl_grid, indexing="ij")
    lifetimes_rec_2d = t_val - wsl_2d  # convert back to model's internal recency

    # Build a grid DataFrame for out-of-sample prediction
    n_pts = freq_2d.size
    grid_df = pd.DataFrame({
        "customer_id": np.arange(n_pts).astype(str),
        "frequency":   freq_2d.ravel().astype(float),
        "recency":     lifetimes_rec_2d.ravel().astype(float),
        "T":           np.full(n_pts, t_val),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = (
            bgm.expected_purchases(data=grid_df, future_t=horizon_weeks)
            .mean(("chain", "draw"))
            .to_series()
            .values
            .reshape(freq_2d.shape)
        )

    fig_fr = px.imshow(
        matrix,
        x=wsl_grid,
        y=freq_grid,
        color_continuous_scale="Blues",
        labels={"x": "Weeks since last purchase", "y": "Frequency", "color": "Expected purchases"},
        title=f"Expected purchases in next {horizon_months}m",
        aspect="auto",
    )
    fig_fr.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
    st.caption(f"Customer tenure (T) held at dataset median ({t_val:.0f} weeks) for visualisation purposes.")
    st.plotly_chart(fig_fr, width="stretch")

with col_d2:
    st.markdown("**Probability alive matrix** — likelihood each customer is still active")
    st.markdown(
        "Customers who purchased recently (left) are most likely still active. "
        "High-frequency customers who haven't purchased in a long time (bottom-right) "
        "are the most clearly churned — the model is near-certain they've dropped off."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alive_matrix = (
            bgm.expected_probability_alive(data=grid_df)
            .mean(("chain", "draw"))
            .to_series()
            .values
            .reshape(freq_2d.shape)
        )

    fig_alive = px.imshow(
        alive_matrix,
        x=wsl_grid,
        y=freq_grid,
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        labels={"x": "Weeks since last purchase", "y": "Frequency", "color": "P(alive)"},
        title="Probability customer is still active",
        aspect="auto",
    )
    fig_alive.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
    st.plotly_chart(fig_alive, width="stretch")

# ── CLV distribution ───────────────────────────────────────────────────────────
st.subheader("CLV Distribution")

col_h1, col_h2 = st.columns(2)

with col_h1:
    clv_max  = summary["clv"].max()
    bin_size = clv_max / 60
    fig_hist = px.histogram(
        summary,
        x="clv",
        title=f"CLV distribution over {horizon_months} months",
        labels={"clv": "Predicted CLV (£)"},
        color_discrete_sequence=["#2C78B7"],
    )
    fig_hist.update_traces(xbins=dict(start=0, end=clv_max * 1.05, size=bin_size))
    fig_hist.update_xaxes(tickprefix="£", tickformat=",")
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, width="stretch")

with col_h2:
    sorted_clv    = np.sort(summary["clv"].values)
    cum_clv       = np.cumsum(sorted_clv) / sorted_clv.sum() * 100
    cum_customers = np.arange(1, len(sorted_clv) + 1) / len(sorted_clv) * 100

    idx_80        = int(0.8 * len(sorted_clv))
    pct_top20     = 100 - cum_clv[idx_80]

    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(
        x=cum_customers, y=cum_clv,
        mode="lines", name="CLV concentration",
        line=dict(color="#B85F3D", width=2.5),
    ))
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines", name="Perfect equality",
        line=dict(color="#9CA3AF", width=1.5, dash="dash"),
    ))
    fig_lorenz.add_annotation(
        x=85, y=30,
        text=f"Top 20% of customers<br>account for {pct_top20:.0f}% of CLV",
        showarrow=False, font=dict(size=12),
        bgcolor="rgba(250,249,245,0.9)", bordercolor="#CEC9BC", borderwidth=1,
    )
    fig_lorenz.update_layout(
        title="CLV Concentration (Lorenz Curve)",
        xaxis_title="Cumulative % of customers (low → high CLV)",
        yaxis_title="Cumulative % of total CLV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_lorenz, width="stretch")

# ── Top customers ──────────────────────────────────────────────────────────────
st.subheader("Top Customers by Predicted CLV")

top_n = st.slider("Show top N customers", min_value=10, max_value=100, value=25, step=5)

top_customers = (
    summary[["customer_id", "frequency", "recency", "T", "monetary_value",
              "predicted_purchases", "prob_alive", "expected_aov", "clv"]]
    .sort_values("clv", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
# Convert model-internal recency to weeks since last purchase for display
top_customers["recency"] = (top_customers["T"] - top_customers["recency"]).clip(lower=0)

pred_col = f"Predicted purchases (next {horizon_months}m)"
clv_col  = f"{horizon_months}-month CLV (£)"

top_customers.columns = [
    "Customer ID", "Repeat purchases", "Weeks since last purchase", "Tenure (wks)",
    "Historical AOV (£)", pred_col, "P(active)", "Expected AOV (£)", clv_col,
]

st.dataframe(
    top_customers,
    width="stretch",
    column_config={
        "Historical AOV (£)": st.column_config.NumberColumn(format="£ %.2f"),
        "Expected AOV (£)":   st.column_config.NumberColumn(format="£ %.2f"),
        clv_col:              st.column_config.NumberColumn(format="£ %.2f"),
        pred_col:             st.column_config.NumberColumn(format="%.2f"),
        "P(active)":          st.column_config.ProgressColumn(format="%.0%", min_value=0, max_value=1),
    },
    hide_index=True,
)

# ── Download ───────────────────────────────────────────────────────────────────
export = summary.copy()
export["weeks_since_last_purchase"] = (export["T"] - export["recency"]).clip(lower=0)
export = export.drop(columns=["recency"])
export.columns = [c.replace("_", " ").title() for c in export.columns]
csv = export.to_csv(index=False)

st.download_button(
    label="⬇️ Download full CLV predictions (CSV)",
    data=csv,
    file_name="clv_predictions.csv",
    mime="text/csv",
)
