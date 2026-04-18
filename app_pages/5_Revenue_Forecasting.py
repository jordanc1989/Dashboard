import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    NEUTRAL_GRID,
    apply_sidebar_filters,
    build_revenue_series,
    load_data,
    render_dataset_subtitle,
    finalize_fig,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


st.set_page_config(
    page_title="Revenue Forecasting · Customer Analytics",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

df = load_data()
df = apply_sidebar_filters(df)

render_dataset_subtitle(df)

st.markdown(
    "Forecasts future revenue from the historical transaction stream. "
    "Two complementary classical models are available: **SARIMA** (Seasonal "
    "ARIMA) and the **Theta method** (M3-competition-winning decomposition "
    "forecaster). A holdout window is reserved at the end of the series to "
    "score out-of-sample accuracy."
)


# ── Controls ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

with c1:
    freq_label = st.selectbox(
        "Frequency",
        ["Weekly", "Monthly"],
        index=0,
        help=(
            "Weekly gives ~109 observations for this dataset — the best "
            "balance of resolution and sample size. Monthly (~25 points) "
            "captures yearly seasonality but with less granularity."
        ),
    )
freq_code = "W" if freq_label == "Weekly" else "MS"
default_season = 52 if freq_code == "W" else 12

with c2:
    model_name = st.selectbox(
        "Model",
        ["SARIMA", "Theta"],
        index=0,
    )

with c3:
    horizon = st.slider(
        "Forecast horizon",
        min_value=1,
        max_value=52 if freq_code == "W" else 12,
        value=13 if freq_code == "W" else 6,
        help="Number of periods to forecast beyond the observed data.",
    )

with c4:
    holdout = st.slider(
        "Holdout periods",
        min_value=0,
        max_value=16 if freq_code == "W" else 4,
        value=8 if freq_code == "W" else 2,
        help="Periods held out at the end of the series for backtest metrics.",
    )


# ── Build series ──────────────────────────────────────────────────────
series = build_revenue_series(df, freq=freq_code).astype(float)

last_date_in_data = df["InvoiceDate"].max()
if freq_code == "W":
    if series.index[-1] > last_date_in_data:
        series = series.iloc[:-1]
elif freq_code == "MS":
    if series.index[-1].to_period("M") == last_date_in_data.to_period("M") and \
            last_date_in_data.day < 28:
        series = series.iloc[:-1]

if len(series) < 8:
    st.warning(
        "Not enough data to fit a meaningful forecast. Try widening the sidebar "
        "date range or switching frequency."
    )
    st.stop()

if holdout >= len(series):
    holdout = max(0, len(series) // 4)

train = series.iloc[: len(series) - holdout] if holdout > 0 else series
test = series.iloc[len(series) - holdout :] if holdout > 0 else series.iloc[0:0]


# ── Model-specific controls ───────────────────────────────────────────
with st.expander("Model parameters", expanded=False):
    if model_name == "SARIMA":
        cc1, cc2, cc3 = st.columns(3)
        p = cc1.number_input(
            "Short-term memory",
            min_value=0, max_value=5, value=1,
            help="How many past weeks/months directly influence the current value. "
                 "Higher = longer memory, but risks overfitting on short series."
        )
        d = cc2.number_input(
            "Trend removal",
            min_value=0, max_value=2, value=1,
            help="Removes the overall upward or downward trend before modelling. "
                 "1 is correct for most revenue series. 0 = no trend removal, 2 = rarely needed."
        )
        q = cc3.number_input(
            "Shock absorption",
            min_value=0, max_value=5, value=1,
            help="How quickly the model 'forgets' a surprise spike or dip. "
                 "1 absorbs last period's shock; higher values remember shocks longer."
        )

        cc4, cc6, cc7 = st.columns(3)
        P = cc4.number_input(
            "Yearly memory",
            min_value=0, max_value=2, value=0,
            help="Like short-term memory, but looking back at the same period in previous years "
                 "(e.g. the same week last year). Usually 0 with limited history."
        )
        Q = cc6.number_input(
            "Yearly shock absorption",
            min_value=0, max_value=2, value=1,
            help="Corrects for mis-forecasting the same seasonal period last year. "
                 "e.g. if the model underestimated last Christmas, it adjusts this year's forecast."
        )
        s = cc7.number_input(
            "Season length",
            min_value=0, max_value=52, value=default_season,
            help="How many periods make up one full seasonal cycle. "
                 "52 for weekly data (one year), 12 for monthly. Set to 0 to disable seasonality."
        )
        st.caption(
            "This model looks for patterns in recent weeks (p, q) and adjusts for the overall trend (d)."
            "It also accounts for the same time of year in previous years (Q, s). "
            "Seasonal differencing is fixed to 0 for this dataset (~2 years), because "
            "that is not enough history for stable seasonal differencing."
        )
    else:
        theta_param = st.slider(
            "θ (theta)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help=(
                "Theta method decomposes the series into two theta-lines. "
                "θ=2 is the classical value. Higher = more weight on short-term "
                "curvature; θ=1 recovers a simple linear trend."
            ),
        )
        theta_period = default_season
        if holdout > 0 and len(train) < 2 * theta_period:
            max_holdout = max(0, len(series) - 2 * theta_period)
            st.warning(
                f"Training window ({len(train)} periods) is shorter than 2 x "
                f"seasonal period ({2 * theta_period}), so the **holdout** forecast "
                f"falls back to trend-only (hence the flat line). The **future** "
                f"forecast still uses seasonality because it fits on the full "
                f"{len(series)} periods.\n\nFor a like-for-like backtest, "
                f"reduce the holdout slider to <= {max_holdout}. "
                f"(Theta seasonal period is automatic: {theta_period} for {freq_label.lower()} data.)"
            )
        st.caption(
            "The Theta method (Assimakopoulos & Nikolopoulos, 2000) was a top "
            "performer in the M3 forecasting competition. It combines simple "
            "exponential smoothing with drift on two decomposed 'theta-lines' "
            "and is fast, robust, and surprisingly hard to beat on business data. "
            f"Deseasonalization uses the model's automatic mode with fixed seasonal period "
            f"{theta_period} ({'weekly yearly' if freq_code == 'W' else 'monthly yearly'})."
        )


# ── Fit / forecast helpers ────────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting SARIMA...")
def fit_sarima(y: pd.Series, order, seasonal_order):
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def sarima_forecast(fitted, n, alpha=0.1):
    fc = fitted.get_forecast(steps=n)
    ci = fc.conf_int(alpha=alpha)
    return fc.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1]


@st.cache_resource(show_spinner="Fitting Theta model...")
def fit_theta(y: pd.Series, period: int):
    deseasonalize = period >= 2 and len(y) >= 2 * period
    kwargs = dict(deseasonalize=deseasonalize)
    if deseasonalize:
        kwargs["period"] = period
        if (y <= 0).any():
            kwargs["method"] = "additive"  # Multiplicative seasonality is undefined for non-positive values.
    model = ThetaModel(y, **kwargs)
    return model.fit(disp=False)


def theta_forecast(fitted, n, theta: float, alpha=0.1):
    mean = fitted.forecast(steps=n, theta=theta)
    try:
        pi = fitted.prediction_intervals(steps=n, theta=theta, alpha=alpha)
        lo, hi = pi.iloc[:, 0], pi.iloc[:, 1]
    except Exception:
        resid_std = float(np.std(np.asarray(fitted.resid)))
        z = 1.645
        steps = np.arange(1, n + 1)
        band = z * resid_std * np.sqrt(steps)
        lo = pd.Series(mean.values - band, index=mean.index)
        hi = pd.Series(mean.values + band, index=mean.index)
    return mean, lo, hi


def _theta_in_sample_fit(y: pd.Series, alpha: float, b0: float) -> pd.Series:
    """One-step-ahead in-sample fit using the Theta method's SES-with-drift recursion.

    ThetaModelResults doesn't expose fittedvalues, so we reconstruct them from
    the fitted parameters: the θ=2 line is forecast via SES and the θ=0 line
    contributes a linear drift of b0/2 per step. Good enough for residual plots.
    """
    if not np.isfinite(alpha) or not np.isfinite(b0):
        return pd.Series(np.nan, index=y.index)

    vals = y.values.astype(float)
    n = len(vals)
    level = np.empty(n)
    level[0] = vals[0]
    for t in range(1, n):
        level[t] = alpha * vals[t] + (1.0 - alpha) * level[t - 1]

    drift = (b0 / 2.0) * np.arange(n)
    fitted = np.concatenate([[np.nan], level[:-1] + drift[:-1] + (b0 / 2.0)])
    return pd.Series(fitted, index=y.index)


def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── Prepare future index ──────────────────────────────────────────────
freq_offset = pd.tseries.frequencies.to_offset(freq_code)
future_index = pd.date_range(
    start=series.index[-1] + freq_offset,
    periods=horizon,
    freq=freq_code,
)


# ── Fit models and produce forecasts ──────────────────────────────────
pred_mean = pred_lo = pred_hi = None
backtest_mape = backtest_rmse = None
future_mean = future_lo = future_hi = None
residuals = None
summary_caption = ""

try:
    if model_name == "SARIMA":
        order = (int(p), int(d), int(q))
        seasonal_order = (int(P), 0, int(Q), int(s)) if s >= 2 else (0, 0, 0, 0)

        if holdout > 0:
            fitted_train = fit_sarima(train, order, seasonal_order)
            pm, plo, phi = sarima_forecast(fitted_train, len(test))
            pred_mean = pd.Series(pm.values, index=test.index)
            pred_lo = pd.Series(plo.values, index=test.index)
            pred_hi = pd.Series(phi.values, index=test.index)
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        full_fit = fit_sarima(series, order, seasonal_order)
        fm, flo, fhi = sarima_forecast(full_fit, horizon)
        future_mean = pd.Series(fm.values, index=future_index)
        future_lo = pd.Series(flo.values, index=future_index)
        future_hi = pd.Series(fhi.values, index=future_index)
        residuals = pd.Series(full_fit.resid).dropna()
        summary_caption = (
            f"AIC: {full_fit.aic:.1f}  ·  BIC: {full_fit.bic:.1f}  ·  "
            f"Residual std: £{residuals.std():,.0f}"
        )
    else:
        period = int(theta_period)

        if holdout > 0:
            fitted_train = fit_theta(train, period)
            pm, plo, phi = theta_forecast(fitted_train, len(test), float(theta_param))
            pred_mean = pd.Series(np.asarray(pm), index=test.index)
            pred_lo = pd.Series(np.asarray(plo), index=test.index)
            pred_hi = pd.Series(np.asarray(phi), index=test.index)
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        full_fit = fit_theta(series, period)
        fm, flo, fhi = theta_forecast(full_fit, horizon, float(theta_param))
        future_mean = pd.Series(np.asarray(fm), index=future_index)
        future_lo = pd.Series(np.asarray(flo), index=future_index)
        future_hi = pd.Series(np.asarray(fhi), index=future_index)

        params = getattr(full_fit, "params", {})
        alpha_est = float(params.get("alpha", float("nan")))
        b0_est = float(params.get("b0", float("nan")))
        fitted_vals = _theta_in_sample_fit(series, alpha_est, b0_est)
        residuals = (series - fitted_vals).dropna()
        summary_caption = (
            f"α: {alpha_est:.3f}  ·  b₀: {b0_est:.2f}  ·  θ: {float(theta_param):.1f}  ·  "
            f"Residual std: £{residuals.std():,.0f}"
        )
except Exception as exc:
    st.error(f"Model failed to fit: {exc}")
    st.stop()


# Revenue is non-negative by definition; clip Gaussian forecast/CIs at 0
future_mean = future_mean.clip(lower=0)
future_lo = future_lo.clip(lower=0)
future_hi = future_hi.clip(lower=0)
if pred_mean is not None:
    pred_mean = pred_mean.clip(lower=0)
    pred_lo = pred_lo.clip(lower=0)
    pred_hi = pred_hi.clip(lower=0)


# ── KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations", f"{len(series):,}")
k2.metric("Holdout MAPE", f"{backtest_mape:.1f}%" if backtest_mape is not None else "—")
k3.metric("Holdout RMSE", f"£{backtest_rmse:,.0f}" if backtest_rmse is not None else "—")
k4.metric(
    f"Next {horizon}-period forecast",
    f"£{float(future_mean.sum()):,.0f}",
)


# ── Main chart ────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train.index, y=train.values,
    name="Training",
    mode="lines",
    line=dict(color="#2C78B7", width=2, shape="spline"),
))

if holdout > 0:
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values,
        name="Holdout actual",
        mode="lines+markers",
        line=dict(color="#141413", width=2, dash="dot"),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=pred_mean.index, y=pred_mean.values,
        name="Holdout forecast",
        mode="lines",
        line=dict(color="#B85F3D", width=2, dash="dash"),
    ))

fig.add_trace(go.Scatter(
    x=list(future_hi.index) + list(future_lo.index[::-1]),
    y=list(future_hi.values) + list(future_lo.values[::-1]),
    fill="toself",
    fillcolor="rgba(184,95,61,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip",
    name="90% CI (future)",
    showlegend=True,
))

fig.add_trace(go.Scatter(
    x=future_mean.index, y=future_mean.values,
    name="Forecast",
    mode="lines",
    line=dict(color="#B85F3D", width=2.5, shape="spline"),
))

fig.update_layout(
    title=f"{freq_label} Revenue — {model_name} Forecast",
    yaxis=dict(
        title="Revenue (£)",
        tickprefix="£",
        tickformat=",",
        rangemode="tozero",
    ),
    hovermode="x unified",
)
finalize_fig(fig, unified_hover=True)
st.plotly_chart(fig, width="stretch")


# ── Residual diagnostics ──────────────────────────────────────────────
with st.expander("Residual diagnostics"):
    d1, d2 = st.columns(2)

    with d1:
        fig_r = go.Figure(go.Scatter(
            x=residuals.index, y=residuals.values,
            mode="lines",
            line=dict(color="#2E7D68", width=1.5),
        ))
        fig_r.add_hline(y=0, line=dict(color=NEUTRAL_GRID, width=1, dash="dot"))
        fig_r.update_layout(
            title="Residuals over time",
            yaxis_title="Residual (£)",
            yaxis_tickprefix="£",
            yaxis_tickformat=",",
            showlegend=False,
        )
        finalize_fig(fig_r, unified_hover=True)
        st.plotly_chart(fig_r, width="stretch")

    with d2:
        fig_h = go.Figure(go.Histogram(
            x=residuals.values,
            nbinsx=20,
            marker=dict(color="#7A52B3"),
        ))
        fig_h.update_layout(
            title="Residual distribution",
            xaxis_title="Residual (£)",
            yaxis_title="Count",
            showlegend=False,
        )
        finalize_fig(fig_h)
        st.plotly_chart(fig_h, width="stretch")

    st.caption(summary_caption)


# ── Forecast table ────────────────────────────────────────────────────
with st.expander("Forecast values"):
    fc_table = pd.DataFrame({
        "Period": future_mean.index.strftime("%Y-%m-%d"),
        "Forecast (£)": future_mean.values.astype(float),
        "Lower 90% (£)": future_lo.values.astype(float),
        "Upper 90% (£)": future_hi.values.astype(float),
    })
    st.dataframe(
        fc_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Forecast (£)": st.column_config.NumberColumn(format="£%.0f"),
            "Lower 90% (£)": st.column_config.NumberColumn(format="£%.0f"),
            "Upper 90% (£)": st.column_config.NumberColumn(format="£%.0f"),
        },
    )


st.caption(
    "Dataset covers ~2 years (Dec 2009 - Dec 2011). Yearly seasonality is "
    "identifiable but rests on only ~2 full cycles. Forecasts more than a "
    "season ahead should be treated as indicative."
)
