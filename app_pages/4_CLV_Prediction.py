import hashlib
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
from pymc_marketing.clv.utils import rfm_train_test_split
from utils import (
    COLOR_SCALE_EXPECTED_PURCHASES,
    COLOR_SCALE_P_ALIVE,
    NEUTRAL_GRID,
    load_data,
    apply_sidebar_filters,
    build_clv_summary,
    render_page_header,
    render_page_footer,
    section,
    finalise_fig,
)

WEEKS_PER_MONTH = 4.345  # 365.25 / 12 / 7
DEFAULT_MAX_CUSTOMERS_FOR_CLV = 2500

def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

st.set_page_config(
    page_title="CLV prediction · Customer analytics",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide"
)

df = load_data()
df = apply_sidebar_filters(df)

render_page_header("clv", df)

with st.expander("How the models work", icon=":material/help_outline:"):
    st.markdown("""
**BG/NBD model** assumes each customer has an unobserved purchase rate (Poisson) and dropout
probability (Geometric). It takes three inputs per customer:

- **Frequency** — number of repeat purchases (total orders minus the first)
- **Recency** — weeks since their last purchase
- **T** — total weeks since their first purchase (customer age)

It outputs expected future transactions (over a chosen time window), and the probability the
customer is still active (i.e. "alive").

**Gamma-Gamma model** takes customers with at least one repeat purchase and models the
distribution of average spend, assuming spend is independent of purchase frequency.
It outputs a corrected, customer-specific estimate of **expected average order value**.

Both models can be fitted via **MAP estimation** (fast, point estimate) or **MCMC sampling**
(slower but enables credible intervals on every prediction).""")

df_customers = df[~df["is_guest"]]

if df_customers["Customer ID"].nunique() < 50:
    st.warning("Not enough registered customers in this selection. Try 'All' countries or widen the date range.")
    st.stop()

summary = build_clv_summary(df_customers)

if len(summary) < 20:
    st.warning("Too few repeat customers to fit the models reliably. Widen the filters.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────────
section("Model controls", eyebrow="Horizon & inference")
with st.container(border=True):
    with st.form("clv_controls"):
        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
        with col_ctrl1:
            horizon_months = st.slider("Prediction horizon (months)", min_value=1, max_value=12, value=3)
        with col_ctrl2:
            annual_rate = st.slider(
                "Annual discount rate (%)",
                min_value=0.0, max_value=10.0, value=3.5, step=0.5,
                help="Cost of capital used to discount future cash flows to present value. 3-5% is typical for stable retailers."
            ) / 100

        with col_ctrl3:
            winsorise_monetary = st.toggle(
                "Winsorise spend (99th pct)",
                value=True,
                help="Clips extreme values at the 99th percentile before fitting the Gamma-Gamma model. Reduces the influence of one-off large orders on everyone's predicted AOV.",
            )
            low_memory_mode = st.toggle(
                "Low-memory mode",
                value=True,
                help="Reduces peak memory by fitting CLV on a capped customer set. Recommended for Streamlit Community Cloud.",
            )
        with col_ctrl4:
            use_mcmc = st.toggle(
                "MCMC sampling",
                value=False,
                help="Full Bayesian inference. Adds 90% credible intervals to every prediction, but can consume substantial RAM.",
            )
            max_customers_for_fit = st.slider(
                "Max customers to model",
                min_value=500,
                max_value=5000,
                step=250,
                value=DEFAULT_MAX_CUSTOMERS_FOR_CLV,
                disabled=not low_memory_mode,
                help="When low-memory mode is on, fit models on the most active customers up to this cap.",
            )
        st.form_submit_button("Apply model settings", type="primary")

monthly_discount = (1 + annual_rate) ** (1 / 12) - 1

if use_mcmc and not st.session_state.get("_mcmc_toast_shown"):
    st.toast("MCMC mode enabled — first run takes ~1-2 minutes. Results are cached once fitted.")
    st.session_state["_mcmc_toast_shown"] = True
if not use_mcmc:
    st.session_state["_mcmc_toast_shown"] = False

horizon_weeks = horizon_months * WEEKS_PER_MONTH
fit_method = "mcmc" if use_mcmc else "map"

fit_summary = summary.copy()
if low_memory_mode and len(fit_summary) > max_customers_for_fit:
    # Keep the most information-rich customers for model fitting.
    fit_summary = (
        fit_summary
        .sort_values(["frequency", "monetary_value", "T"], ascending=False)
        .head(max_customers_for_fit)
        .copy()
    )
    st.caption(
        f"Low-memory mode active: fitting on {len(fit_summary):,} of {len(summary):,} customers."
    )

# ── Fit models ────────────────
bg_data = fit_summary[fit_summary["frequency"] > 0][["customer_id", "frequency", "recency", "T"]].copy()
gg_data = fit_summary[(fit_summary["frequency"] > 0) & (fit_summary["monetary_value"] > 0)][["customer_id", "frequency", "monetary_value"]].copy()

if winsorise_monetary:
    cap = gg_data["monetary_value"].quantile(0.99)
    gg_data["monetary_value"] = gg_data["monetary_value"].clip(upper=cap)


@st.cache_resource(max_entries=4)
def fit_bgnbd(data_hash: str, method: str, _data: pd.DataFrame):
    bgm = BetaGeoModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_kwargs = {"method": method, "data": _data}
        if method == "mcmc":
            try:
                import nutpie  # noqa: F401
                fit_kwargs["nuts_sampler"] = "nutpie"
            except ImportError:
                pass
        bgm.fit(**fit_kwargs)
    if method == "mcmc":
        bgm = bgm.thin_fit_result(keep_every=2)  # Reduces memory / prediction cost at the expense of slightly less smooth posteriors.
    return bgm


@st.cache_resource(max_entries=4)
def fit_gg(data_hash: str, method: str, _data: pd.DataFrame):
    ggm = GammaGammaModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_kwargs = {"method": method, "data": _data}
        if method == "mcmc":
            try:
                import nutpie  # noqa: F401
                fit_kwargs["nuts_sampler"] = "nutpie"
            except ImportError:
                pass
        ggm.fit(**fit_kwargs)
    if method == "mcmc":
        ggm = ggm.thin_fit_result(keep_every=2)
    return ggm


spinner_msg = (
    "Fitting BG/NBD and Gamma-Gamma models via MCMC — this may take 1-2 minutes…"
    if fit_method == "mcmc" else
    "Fitting BG/NBD and Gamma-Gamma models (MAP)…"
)
with st.spinner(spinner_msg):
    bgm = fit_bgnbd(_df_hash(bg_data), fit_method, bg_data)
    ggm = fit_gg(_df_hash(gg_data), fit_method, gg_data)

# ── Gamma-Gamma independence assumption check ────────────
fm_corr = gg_data[["frequency", "monetary_value"]].corr().iloc[0, 1]
if abs(fm_corr) > 0.3:
    st.warning(
        f"Warning: Frequency-monetary correlation is **{fm_corr:+.2f}**. "
        "The Gamma-Gamma model assumes these are approximately independent "
        "(|r| < 0.3). Spend predictions may be biased for this segment."
    )
else:
    st.caption(
        f"OK — Gamma-Gamma independence assumption holds "
        f"(frequency-monetary correlation = {fm_corr:+.2f}, |r| < 0.3)."
    )

# ── Predictions ────────────────────────────────────────────────────────────────
# Label each series with customer_id before assigning back
predicted_purchases = (
    bgm.expected_purchases(data=bg_data, future_t=horizon_weeks)
    .mean(("chain", "draw"))
    .to_series()
    .set_axis(bg_data["customer_id"].values) 
)
prob_alive = (
    bgm.expected_probability_alive(data=bg_data)
    .mean(("chain", "draw"))
    .to_series()
    .set_axis(bg_data["customer_id"].values)
)
expected_aov = (
    ggm.expected_customer_spend(data=gg_data)
    .mean(("chain", "draw"))
    .to_series()
    .set_axis(gg_data["customer_id"].values)   # gg_data is a subset — freq > 0 only
)

# GammaGamma CLV is only defined for repeat buyers with positive spend.
clv_input = fit_summary[(fit_summary["frequency"] > 0) & (fit_summary["monetary_value"] > 0)].copy()
clv_da = ggm.expected_customer_lifetime_value(
    transaction_model=bgm,
    data=clv_input,
    future_t=horizon_months,
    discount_rate=monthly_discount,
    time_unit="W",
)
clv = (
    clv_da.mean(("chain", "draw"))
    .to_series()
    .set_axis(clv_input["customer_id"].values)
    .clip(lower=0)
)

summary = summary.set_index("customer_id")
summary["predicted_purchases"] = predicted_purchases.reindex(summary.index)
summary["prob_alive"]           = prob_alive.reindex(summary.index)
summary["expected_aov"]         = expected_aov.reindex(summary.index).fillna(0)  # freq=0 customers have no AOV estimate
summary["clv"]                  = clv.reindex(summary.index).fillna(0)

if use_mcmc:
    ci = clv_da.quantile([0.05, 0.95], dim=["chain", "draw"])
    clv_p05 = (
        ci.sel(quantile=0.05).to_series()
        .set_axis(clv_input["customer_id"].values)
        .clip(lower=0)
    )
    clv_p95 = (
        ci.sel(quantile=0.95).to_series()
        .set_axis(clv_input["customer_id"].values)
        .clip(lower=0)
    )
    summary["clv_p05"] = clv_p05.reindex(summary.index).fillna(0)
    summary["clv_p95"] = clv_p95.reindex(summary.index).fillna(0)

summary = summary.reset_index()  # restore customer_id as a plain column

# ── KPIs ─────────────────────────────────────────
st.space("small")
section("Headline CLV", eyebrow=f"Next {horizon_months}-month horizon")

total_clv = summary["clv"].sum()
median_clv = clv.median()  # median over modelled customers only; summary includes zero-filled one-time buyers

with st.container(horizontal=True):
    st.metric("Customers modelled", f"{len(summary):,}", border=True)
    if use_mcmc:
        st.metric(
            f"Expected revenue (next {horizon_months}m)",
            f"£{total_clv:,.0f}",
            border=True,
            help=f"90% credible interval: £{summary['clv_p05'].sum():,.0f} - £{summary['clv_p95'].sum():,.0f}",
        )
        st.metric(
            "Median CLV",
            f"£{median_clv:,.2f}",
            border=True,
            help=f"90% credible interval: £{summary['clv_p05'].median():,.2f} - £{summary['clv_p95'].median():,.2f}",
        )
    else:
        st.metric(
            f"Expected revenue (next {horizon_months}m)",
            f"£{total_clv:,.0f}",
            border=True,
        )
        st.metric("Median CLV", f"£{median_clv:,.2f}", border=True)
    st.metric(
        "Median P(still active)",
        f"{summary['prob_alive'].median()*100:.1f}%",
        border=True,
    )

# ── Out-of-sample validation ────────────────
@st.cache_data
def run_holdout_validation(data_hash: str, _df: pd.DataFrame) -> tuple | None:
    invoices = (
        _df.groupby(["Customer ID", "Invoice", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )

    obs_start = _df["InvoiceDate"].min()
    obs_end   = _df["InvoiceDate"].max()
    cal_end   = obs_start + pd.Timedelta(days=int((obs_end - obs_start).days * 0.75))
    holdout_weeks = (obs_end - cal_end).days / 7.0

    ch = rfm_train_test_split(
        invoices,
        customer_id_col="Customer ID",
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


st.space("small")
section("Out-of-sample validation", eyebrow="75 / 25 time split")
with st.expander(
    "BG/NBD model: predicted vs actual purchases",
    expanded=False,
    icon=":material/fact_check:",
):
    st.markdown(
        "Model performance is evaluated by withholding the last 25% of the observation window, "
        "fitting BG/NBD only on the earlier 75% (via MAP) and predicting the number of purchases each "
        "customer will make in the holdout. The chart bins customers by their frequency in the "
        "calibration period and plots predicted vs. actual mean purchases in the holdout. "
        "Close alignment with the dashed y=x line indicates the model is well-calibrated. "
    )

    validation = run_holdout_validation(_df_hash(df_customers), df_customers)
    if validation is None:
        st.caption("Not enough data to run holdout validation for this selection.")
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
            line=dict(color=NEUTRAL_GRID, width=1.5, dash="dash"),
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
        finalise_fig(fig_val)
        st.plotly_chart(fig_val, width="stretch")


# ── GG out-of-sample validation ─────────────
@st.cache_data
def run_gg_holdout_validation(data_hash: str, _df: pd.DataFrame) -> tuple | None:
    invoices = (
        _df.groupby(["Customer ID", "Invoice", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )

    obs_start = _df["InvoiceDate"].min()
    obs_end   = _df["InvoiceDate"].max()
    cal_end   = obs_start + pd.Timedelta(days=int((obs_end - obs_start).days * 0.75))

    ch = rfm_train_test_split(
        invoices,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        train_period_end=cal_end,
        test_period_end=obs_end,
        time_unit="W",
    )

    ch_gg = ch[(ch["frequency"] > 0) & (ch["monetary_value"] > 0)].copy()

    # Actual mean spend per transaction in holdout for customers who purchased in both periods
    holdout_spend = (
        invoices[invoices["InvoiceDate"] > cal_end]
        .groupby("Customer ID")["Revenue"]
        .mean()
        .reset_index()
        .rename(columns={"Customer ID": "customer_id", "Revenue": "actual_holdout_spend"})
    )
    ch_gg = ch_gg.merge(holdout_spend, on="customer_id", how="inner")

    if len(ch_gg) < 20:
        return None

    cal_gg = ch_gg[["customer_id", "frequency", "monetary_value"]].copy()
    ggm_cal = GammaGammaModel(data=cal_gg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ggm_cal.fit(method="map")

    ch_gg["predicted_spend"] = (
        ggm_cal.expected_customer_spend(data=cal_gg)
        .mean(("chain", "draw"))
        .to_series()
        .values
    )

    return ch_gg, cal_end


with st.expander(
    "Gamma-Gamma spend model: predicted vs actual AOV",
    expanded=False,
    icon=":material/fact_check:",
):
    st.markdown(
        "The Gamma-Gamma model is validated using the same 75/25 time split. "
        "It is fitted on calibration-period transactions (using MAP) and its predicted average order value "
        "is compared against each customer's actual mean spend in the holdout period. "
        "Only customers who purchased in both periods are included since we need actuals to "
        "compare against. The closer points cluster around the dashed y=x line, the better "
        "the model's spend predictions."
    )

    gg_validation = run_gg_holdout_validation(_df_hash(df_customers), df_customers)
    if gg_validation is None:
        st.caption("Not enough data to run Gamma-Gamma holdout validation for this selection.")
    else:
        ch_gg, gg_cal_end = gg_validation

        mae_gg  = (ch_gg["predicted_spend"] - ch_gg["actual_holdout_spend"]).abs().mean()
        rmse_gg = np.sqrt(((ch_gg["predicted_spend"] - ch_gg["actual_holdout_spend"]) ** 2).mean())
        r_gg    = ch_gg[["predicted_spend", "actual_holdout_spend"]].corr().iloc[0, 1]

        gv1, gv2, gv3, gv4 = st.columns(4)
        gv1.metric("Customers compared", f"{len(ch_gg):,}")
        gv2.metric("MAE", f"£{mae_gg:,.2f}")
        gv3.metric("RMSE", f"£{rmse_gg:,.2f}")
        gv4.metric("Pearson r", f"{r_gg:.3f}")

        spend_cap = max(
            ch_gg["predicted_spend"].quantile(0.99),
            ch_gg["actual_holdout_spend"].quantile(0.99),
        )
        plot_gg   = ch_gg[
            (ch_gg["predicted_spend"] <= spend_cap) &
            (ch_gg["actual_holdout_spend"] <= spend_cap)
        ]
        n_clipped = len(ch_gg) - len(plot_gg)

        fig_gg_val = go.Figure()
        fig_gg_val.add_trace(go.Scatter(
            x=[0, spend_cap], y=[0, spend_cap],
            mode="lines", name="Perfect prediction",
            line=dict(color=NEUTRAL_GRID, width=1.5, dash="dash"),
        ))
        fig_gg_val.add_trace(go.Scatter(
            x=plot_gg["predicted_spend"],
            y=plot_gg["actual_holdout_spend"],
            mode="markers",
            name="Customers",
            marker=dict(color="#2E7D68", size=5, opacity=0.5),
            hovertemplate="Predicted AOV: £%{x:,.2f}<br>Actual holdout spend: £%{y:,.2f}<extra></extra>",
        ))
        fig_gg_val.update_layout(
            title=f"Predicted vs actual mean spend (train to {gg_cal_end.date()})",
            xaxis=dict(title="Predicted AOV (£)", tickprefix="£", tickformat=","),
            yaxis=dict(title="Actual holdout mean spend (£)", tickprefix="£", tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        finalise_fig(fig_gg_val)
        st.plotly_chart(fig_gg_val, width="stretch")
        if n_clipped > 0:
            st.caption(
                f"{n_clipped} customer{'s' if n_clipped > 1 else ''} above the 99th percentile "
                "not shown."
            )

# ── Model diagnostics ──────────────────────────────────────────────────────────
st.space("small")
section("Model diagnostics", eyebrow="Frequency × recency surfaces")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("**Frequency-Recency matrix** — expected purchases in the next period")
    st.markdown(
        "Frequent buyers who purchased recently (bottom-left) are predicted to buy most. "
        "Frequent buyers who haven't purchased in a long time (bottom-right) "
        "are likely churned and the model should predict far fewer future purchases."
    )

    t_val    = float(summary["T"].median())
    max_freq = min(int(summary["frequency"].quantile(0.95)), 40)
    max_wsl  = int(t_val)

    freq_grid = np.arange(1, max_freq + 1)
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
        color_continuous_scale=COLOR_SCALE_EXPECTED_PURCHASES,
        labels={"x": "Weeks since last purchase", "y": "Frequency", "color": "Expected purchases"},
        title=f"Expected purchases in next {horizon_months}m",
        aspect="auto",
    )
    fig_fr.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
    st.caption(f"Customer tenure (T) held at dataset median ({t_val:.0f} weeks) for visualisation purposes.")
    st.caption(
        "Scale: lighter = fewer expected purchases in the horizon; darker blue = more "
        "(brand-aligned sequential scale)."
    )
    finalise_fig(fig_fr)
    st.plotly_chart(fig_fr, width="stretch")

with col_d2:
    st.markdown("**Probability alive matrix**: likelihood each customer is still active")
    st.markdown(
        "Customers who purchased recently (left) are most likely still active. "
        "High-frequency customers who haven't purchased in a long time (bottom-right) "
        "are the most churned — the model should be certain they've dropped off."
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
        color_continuous_scale=COLOR_SCALE_P_ALIVE,
        range_color=[0, 1],
        labels={"x": "Weeks since last purchase", "y": "Frequency", "color": "P(alive)"},
        title="Probability customer is still active",
        aspect="auto",
    )
    fig_alive.update_coloraxes(colorbar=dict(thickness=12, len=0.8))
    st.caption(
        "Scale: brown = lower P(active); neutral mid-tone; green = higher P(active). "
        "Same frequency × recency grid as the chart on the left."
    )
    finalise_fig(fig_alive)
    st.plotly_chart(fig_alive, width="stretch")

# ── CLV distribution ───────────────────────────────────────────────────────────
st.space("small")
section("CLV distribution", eyebrow="Histogram & concentration")

col_h1, col_h2 = st.columns(2)

with col_h1:
    x_cap        = summary["clv"].quantile(0.99)
    n_above      = (summary["clv"] > x_cap).sum()
    bin_size     = x_cap / 50
    fig_hist = px.histogram(
        summary[summary["clv"] <= x_cap],
        x="clv",
        title=f"CLV distribution over {horizon_months} months",
        labels={"clv": "Predicted CLV (£)"},
        color_discrete_sequence=["#B85F3D"],
    )
    fig_hist.update_traces(xbins=dict(start=0, end=x_cap * 1.02, size=bin_size))
    fig_hist.update_xaxes(tickprefix="£", tickformat=",")
    fig_hist.update_layout(showlegend=False)
    finalise_fig(fig_hist)
    st.plotly_chart(fig_hist, width="stretch")
    if n_above > 0:
        st.caption(f"{n_above} customer{'s' if n_above > 1 else ''} above £{x_cap:,.0f} (99th percentile) not shown.")

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
        line=dict(color=NEUTRAL_GRID, width=1.5, dash="dash"),
    ))
    fig_lorenz.add_annotation(
        x=85, y=30,
        text=f"Top 20% of customers<br>account for {pct_top20:.0f}% of CLV",
        showarrow=False, font=dict(size=12),
        bgcolor="rgba(250,249,245,0.9)", bordercolor="#CEC9BC", borderwidth=1,
    )
    fig_lorenz.update_layout(
            title="CLV concentration (Lorenz curve)",
        xaxis_title="Cumulative % of customers (low → high CLV)",
        yaxis_title="Cumulative % of total CLV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    finalise_fig(fig_lorenz, unified_hover=True)
    st.plotly_chart(fig_lorenz, width="stretch")

# ── Top customers ───────────────
st.space("small")
section("Top customers by predicted CLV", eyebrow="Targeting shortlist")

top_n = st.slider("Show top N customers", min_value=10, max_value=100, value=25, step=5)

base_cols = ["customer_id", "frequency", "recency", "T", "monetary_value",
             "predicted_purchases", "prob_alive", "expected_aov", "clv"]
if use_mcmc:
    base_cols += ["clv_p05", "clv_p95"]

top_customers = (
    summary[base_cols]
    .sort_values("clv", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
# Convert model-internal recency to weeks since last purchase for display
top_customers["recency"] = (top_customers["T"] - top_customers["recency"]).clip(lower=0)

pred_col = f"Predicted purchases (next {horizon_months}m)"
clv_col  = f"{horizon_months}-month CLV (£)"

col_names = [
    "Customer ID", "Repeat purchases", "Weeks since last purchase", "Tenure (wks)",
    "Historical AOV (£)", pred_col, "P(active)", "Expected AOV (£)", clv_col,
]
if use_mcmc:
    col_names += ["CLV 5th pct (£)", "CLV 95th pct (£)"]

top_customers.columns = col_names

col_cfg = {
    "Historical AOV (£)": st.column_config.NumberColumn(format="£ %.2f"),
    "Expected AOV (£)":   st.column_config.NumberColumn(format="£ %.2f"),
    clv_col:              st.column_config.NumberColumn(format="£ %.2f"),
    pred_col:             st.column_config.NumberColumn(format="%.2f"),
    "P(active)":          st.column_config.ProgressColumn(min_value=0, max_value=1),
}
if use_mcmc:
    col_cfg["CLV 5th pct (£)"]  = st.column_config.NumberColumn(format="£ %.2f")
    col_cfg["CLV 95th pct (£)"] = st.column_config.NumberColumn(format="£ %.2f")

st.dataframe(
    top_customers,
    width="stretch",
    column_config=col_cfg,
    hide_index=True,
)

# ── Download ───────────────
st.space("small")
section("Export", eyebrow="Download")
export = summary.copy()
export["weeks_since_last_purchase"] = (export["T"] - export["recency"]).clip(lower=0)
export = export.drop(columns=["recency"])
export.columns = [c.replace("_", " ").title() for c in export.columns]
if use_mcmc:
    export = export.rename(columns={"Clv P05": "CLV 5th Pct (£)", "Clv P95": "CLV 95th Pct (£)"})
csv = export.to_csv(index=False)

st.download_button(
    label="Download full CLV predictions (CSV)",
    data=csv,
    file_name="clv_predictions.csv",
    mime="text/csv",
    icon=":material/download:",
)

render_page_footer(df, note="Lifetime value · BG/NBD + Gamma-Gamma")
