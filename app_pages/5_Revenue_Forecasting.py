import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    NEUTRAL_GRID,
    apply_sidebar_filters,
    build_revenue_series,
    load_data,
    render_page_header,
    section,
    finalise_fig,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


st.set_page_config(
    page_title="Revenue forecasting",
    page_icon="static/jordan_cheney_logo_new.png",
    layout="wide",
)

df = load_data()
df = apply_sidebar_filters(df)

render_page_header("forecast", df)


# Controls
section("Forecast controls", eyebrow="Frequency, model, horizon")
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

    with c1:
        freq_label = st.selectbox(
            "Frequency",
            ["Weekly", "Monthly"],
            index=0,
            help=(
                "Weekly gives the best balance of resolution and sample size. Monthly "
                "captures yearly seasonality but with less granularity."
            ),
        )
    freq_code = "W" if freq_label == "Weekly" else "MS"
    default_season = 52 if freq_code == "W" else 12

    with c2:
        model_name = st.selectbox(
            "Model",
            ["SARIMA", "Theta", "ETS", "Seasonal naive"],
            index=0,
            help=(
                "SARIMA and Theta are the workhorses; ETS (Holt-Winters) adds a "
                "second seasonal model; Seasonal naive ('same period last year') is "
                "the baseline every other model is scored against."
            ),
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

    with c5:
        confidence_level = st.slider(
            "Confidence level",
            min_value=80,
            max_value=99,
            value=90,
            step=1,
            help="Width of the forecast prediction interval. Higher = wider band "
            "but more likely to contain the true value.",
        )

alpha = (100 - confidence_level) / 100


# Build series
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


# Model-specific controls
with st.expander(
    "Model parameters",
    expanded=False,
    icon=":material/tune:",
):
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
                 "1 is appropriate for most revenue series. 0 = no trend removal, 2 = rarely needed."
        )
        q = cc3.number_input(
            "Shock absorption",
            min_value=0, max_value=5, value=1,
            help="How quickly the model 'forgets' a surprise spike or dip. "
                 "1 absorbs last period's shock, higher values remember shocks for longer."
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
            "Seasonal differencing is fixed to 0 for this dataset (~2 years), because "
            "that is not enough history for stable seasonal differencing."
        )
    elif model_name == "Theta":
        theta_param = st.slider(
            "θ (theta)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help=(
                "Theta method decomposes the series into two theta-lines. "
                "θ=2 is the classical value. Higher = more weight on short-term "
                "curvature, θ=1 recovers a simple linear trend."
            ),
        )
        theta_period = default_season
        if holdout > 0 and len(train) < 2 * theta_period:
            st.info(
                f"Training window ({len(train)} periods) is shorter than two full "
                f"seasonal cycles ({2 * theta_period}), so the model's built-in "
                f"deseasonalisation can't run on the backtest. The **holdout** forecast "
                f"reconstructs the seasonal shape from training-period data only "
                f"(no look-ahead); the **future** forecast fits the full "
                f"{len(series)} periods and uses the model's native deseasonalisation."
            )
        st.caption(
            "The Theta method (Assimakopoulos & Nikolopoulos, 2000) was a top "
            "performer in the M3 forecasting competition and combines simple "
            "exponential smoothing with drift on two decomposed 'theta-lines' "
            "and is fast, robust and effective on business data. "
            f"The future forecast deseasonalises with the model's automatic mode at a fixed "
            f"seasonal period of {theta_period} "
            f"({'weekly-yearly' if freq_code == 'W' else 'monthly-yearly'}); short backtest "
            f"windows reconstruct the seasonal shape from training data only."
        )
    elif model_name == "ETS":
        ets_damped = st.toggle(
            "Damped trend",
            value=False,
            help="Flattens the trend over the horizon instead of extrapolating it "
            "linearly. Safer for longer horizons; often improves accuracy on "
            "business series that plateau.",
        )
        if holdout > 0 and len(train) < 2 * default_season:
            st.info(
                f"Training window ({len(train)} periods) is shorter than two full "
                f"seasonal cycles ({2 * default_season}), so Holt-Winters can't fit its "
                f"seasonal component on the backtest. The **holdout** forecast reconstructs "
                f"the seasonal shape from training-period data only (no look-ahead); the "
                f"**future** forecast fits the full {len(series)} periods with native "
                f"seasonality."
            )
        st.caption(
            "ETS (error-trend-seasonal) is additive Holt-Winters exponential "
            "smoothing: it tracks level, a linear (optionally damped) trend and an "
            f"additive seasonal cycle of {default_season} "
            f"({'weekly-yearly' if freq_code == 'W' else 'monthly-yearly'}). A natural "
            "companion to Theta, with native prediction intervals."
        )
    else:  # Seasonal naive
        st.caption(
            "Seasonal naive forecasts each period as the same period one cycle ago "
            f"('same {'week' if freq_code == 'W' else 'month'} last year', period "
            f"{default_season}), carried forward. With strong yearly seasonality this is "
            "a deliberately simple but hard-to-beat **baseline** — every other model's "
            "backtest is reported against it. Prediction intervals widen with the square "
            "root of the horizon from the in-sample seasonal-difference spread."
        )


# Fit / forecast helpers
@st.cache_resource(show_spinner="Fitting SARIMA...")
def fit_sarima(y: pd.Series, order: tuple, seasonal_order: tuple):
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
    kwargs = {}
    kwargs["deseasonalize"] = deseasonalize
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
        from scipy.stats import norm
        resid_std = float(np.std(np.asarray(fitted.resid)))
        z = float(norm.ppf(1 - alpha / 2))
        steps = np.arange(1, n + 1)
        band = z * resid_std * np.sqrt(steps)
        lo = pd.Series(mean.values - band, index=mean.index)
        hi = pd.Series(mean.values + band, index=mean.index)
    return mean, lo, hi


def _seasonal_positions(index, freq_code):
    """Within-cycle position per timestamp: ISO week (weekly) or month (monthly)."""
    if freq_code == "W":
        return index.isocalendar().week.to_numpy()
    return index.month.to_numpy()


def _seasonal_factors(train, freq_code):
    """Additive per-cycle seasonal deviations, estimated from training data only."""
    pos = _seasonal_positions(train.index, freq_code)
    seasonal = (
        pd.Series(train.to_numpy() - train.mean(), index=pos)
        .groupby(level=0)
        .mean()
    )
    seasonal = seasonal - seasonal.mean()  # centre: keep only the seasonal shape
    return {int(k): float(v) for k, v in seasonal.items()}


def _seasonal_reconstruct(train, freq_code, n, future_index, trend_forecast):
    """Deseasonalise (training-only additive factors), forecast the trend, reseasonalise.

    `trend_forecast(deseason_series, n)` must return (mean, lo, hi) for the
    deseasonalised series. Used when a window is too short for a model's native
    deseasonalisation (< 2 full cycles). Additive (not multiplicative) so
    zero-revenue periods don't blow the deseasonalised series up; no look-ahead,
    since the seasonal factors come from the training window only.
    """
    factors = _seasonal_factors(train, freq_code)
    s_in = np.array(
        [factors.get(int(p), 0.0) for p in _seasonal_positions(train.index, freq_code)]
    )
    deseason = pd.Series(train.to_numpy() - s_in, index=train.index)
    base_mean, base_lo, base_hi = trend_forecast(deseason, n)
    s_out = np.array(
        [factors.get(int(p), 0.0) for p in _seasonal_positions(future_index, freq_code)]
    )
    return (
        pd.Series(np.asarray(base_mean) + s_out, index=future_index),
        pd.Series(np.asarray(base_lo) + s_out, index=future_index),
        pd.Series(np.asarray(base_hi) + s_out, index=future_index),
    )


def theta_seasonal_backtest(train, freq_code, theta: float, n, future_index, alpha=0.1):
    """Theta backtest with a training-only seasonal reconstruction.

    statsmodels can only deseasonalise a window spanning >= 2 full cycles, which a
    holdout split on this ~2-year series drops below — leaving a flat, trend-only
    backtest. Reconstruct seasonality from training data instead so the backtest
    tracks the cycle (see `_seasonal_reconstruct`).
    """
    def trend_forecast(deseason, k):
        fitted = ThetaModel(deseason, deseasonalize=False).fit(disp=False)
        return theta_forecast(fitted, k, theta, alpha=alpha)

    return _seasonal_reconstruct(train, freq_code, n, future_index, trend_forecast)


def seasonal_naive_forecast(y, m, n, future_index, alpha=0.1):
    """'Same period one cycle ago', carried forward.

    The point forecast repeats the most recent seasonal cycle. Prediction
    intervals widen with sqrt(k), k = cycles ahead, scaled by the in-sample
    seasonal-difference spread (Hyndman & Athanasopoulos, FPP).
    """
    from scipy.stats import norm

    vals = y.to_numpy()
    last_cycle = vals[-m:]
    fc = np.array([last_cycle[(h - 1) % m] for h in range(1, n + 1)])

    diffs = vals[m:] - vals[:-m]
    sigma = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    z = float(norm.ppf(1 - alpha / 2))
    k = np.floor((np.arange(1, n + 1) - 1) / m) + 1
    band = z * sigma * np.sqrt(k)
    return (
        pd.Series(fc, index=future_index),
        pd.Series(fc - band, index=future_index),
        pd.Series(fc + band, index=future_index),
    )


@st.cache_resource(show_spinner="Fitting ETS...")
def fit_ets_native(y: pd.Series, m: int, damped: bool):
    """Holt-Winters with additive trend + additive seasonality (needs >= 2 cycles)."""
    return ETSModel(
        y,
        error="add",
        trend="add",
        damped_trend=damped,
        seasonal="add",
        seasonal_periods=m,
        initialization_method="estimated",
    ).fit(disp=False)


def ets_forecast(y, freq_code, m, n, future_index, alpha=0.1, damped=False):
    """ETS (Holt-Winters) forecast.

    Uses native additive seasonality when the window covers >= 2 cycles; otherwise
    a training-only seasonal reconstruction around a linear/damped ETS trend.
    Returns (mean, lo, hi, fitted_or_None) — the fit is None on the reconstruction
    path (no single seasonal model to read diagnostics from).
    """
    if m >= 2 and len(y) >= 2 * m:
        fit = fit_ets_native(y, m, damped)
        sf = fit.get_prediction(start=len(y), end=len(y) + n - 1).summary_frame(alpha=alpha)
        return (
            pd.Series(sf["mean"].to_numpy(), index=future_index),
            pd.Series(sf["pi_lower"].to_numpy(), index=future_index),
            pd.Series(sf["pi_upper"].to_numpy(), index=future_index),
            fit,
        )

    def trend_forecast(deseason, k):
        fit = ETSModel(
            deseason,
            error="add",
            trend="add",
            damped_trend=damped,
            initialization_method="estimated",
        ).fit(disp=False)
        sf = fit.get_prediction(
            start=len(deseason), end=len(deseason) + k - 1
        ).summary_frame(alpha=alpha)
        return sf["mean"].to_numpy(), sf["pi_lower"].to_numpy(), sf["pi_upper"].to_numpy()

    mean, lo, hi = _seasonal_reconstruct(y, freq_code, n, future_index, trend_forecast)
    return mean, lo, hi, None


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


# Prepare future index
freq_offset = pd.tseries.frequencies.to_offset(freq_code)
future_index = pd.date_range(
    start=series.index[-1] + freq_offset,
    periods=horizon,
    freq=freq_code,
)


# ETS and Seasonal naive both need at least one full seasonal cycle of history.
if model_name in ("ETS", "Seasonal naive") and len(series) <= default_season:
    st.warning(
        f"{model_name} needs more than one full seasonal cycle "
        f"({default_season} periods); this selection has only {len(series)}. "
        "Widen the date range, switch to Monthly, or use SARIMA / Theta."
    )
    st.stop()


# Fit models and produce forecasts
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
            pm, plo, phi = sarima_forecast(fitted_train, len(test), alpha=alpha)
            pred_mean = pd.Series(pm.values, index=test.index)
            pred_lo = pd.Series(plo.values, index=test.index)
            pred_hi = pd.Series(phi.values, index=test.index)
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        full_fit = fit_sarima(series, order, seasonal_order)
        fm, flo, fhi = sarima_forecast(full_fit, horizon, alpha=alpha)
        future_mean = pd.Series(fm.values, index=future_index)
        future_lo = pd.Series(flo.values, index=future_index)
        future_hi = pd.Series(fhi.values, index=future_index)
        residuals = pd.Series(full_fit.resid).dropna()
        summary_caption = (
            f"AIC: {full_fit.aic:.1f}  ·  BIC: {full_fit.bic:.1f}  ·  "
            f"Residual std: £{residuals.std():,.0f}"
        )
    elif model_name == "Theta":
        period = int(theta_period)

        if holdout > 0:
            # When the training window is too short for the model's built-in
            # deseasonalisation (< 2 full cycles), reconstruct seasonality from
            # training data only so the backtest isn't a flat trend-only line.
            if period >= 2 and len(train) < 2 * period:
                pred_mean, pred_lo, pred_hi = theta_seasonal_backtest(
                    train, freq_code, float(theta_param), len(test), test.index, alpha=alpha
                )
            else:
                fitted_train = fit_theta(train, period)
                pm, plo, phi = theta_forecast(fitted_train, len(test), float(theta_param), alpha=alpha)
                pred_mean = pd.Series(np.asarray(pm), index=test.index)
                pred_lo = pd.Series(np.asarray(plo), index=test.index)
                pred_hi = pd.Series(np.asarray(phi), index=test.index)
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        full_fit = fit_theta(series, period)
        fm, flo, fhi = theta_forecast(full_fit, horizon, float(theta_param), alpha=alpha)
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
    elif model_name == "ETS":
        m = int(default_season)

        if holdout > 0:
            pred_mean, pred_lo, pred_hi, _ = ets_forecast(
                train, freq_code, m, len(test), test.index, alpha=alpha, damped=ets_damped
            )
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        future_mean, future_lo, future_hi, ets_fit = ets_forecast(
            series, freq_code, m, horizon, future_index, alpha=alpha, damped=ets_damped
        )
        if ets_fit is not None:
            residuals = pd.Series(ets_fit.resid).dropna()
            summary_caption = (
                f"AIC: {ets_fit.aic:.1f}  ·  "
                f"Trend: {'damped' if ets_damped else 'linear'}  ·  "
                f"Residual std: £{residuals.std():,.0f}"
            )
        else:
            residuals = pd.Series(dtype=float)
            summary_caption = (
                "Seasonal shape reconstructed from a short series — native fit "
                "diagnostics unavailable."
            )
    else:  # Seasonal naive
        m = int(default_season)

        if holdout > 0:
            pred_mean, pred_lo, pred_hi = seasonal_naive_forecast(
                train, m, len(test), test.index, alpha=alpha
            )
            backtest_mape = mape(test.values, pred_mean.values)
            backtest_rmse = rmse(test.values, pred_mean.values)

        future_mean, future_lo, future_hi = seasonal_naive_forecast(
            series, m, horizon, future_index, alpha=alpha
        )
        # Residuals = in-sample seasonal differences (y_t - y_{t-m}).
        seasonal_resid = series.to_numpy()[m:] - series.to_numpy()[:-m]
        residuals = pd.Series(seasonal_resid, index=series.index[m:]).dropna()
        summary_caption = (
            f"Seasonal period: {m}  ·  "
            f"Seasonal-difference residual std: £{residuals.std():,.0f}"
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


# Seasonal-naive benchmark — scores every model against the obvious baseline so
# the backtest answers "does this model beat 'same period last year'?".
naive_mape = naive_rmse = None
_m_bench = int(default_season)
if holdout > 0 and len(train) > _m_bench:
    nb_mean, _, _ = seasonal_naive_forecast(train, _m_bench, len(test), test.index, alpha=alpha)
    nb_mean = nb_mean.clip(lower=0)
    naive_mape = mape(test.values, nb_mean.values)
    naive_rmse = rmse(test.values, nb_mean.values)


# KPIs
st.space("small")
section("Forecast performance", eyebrow="Backtest & outlook")

with st.container(horizontal=True):
    st.metric("Observations", f"{len(series):,}", border=True)
    st.metric(
        "Holdout MAPE",
        f"{backtest_mape:.1f}%" if backtest_mape is not None else "-",
        border=True,
    )
    st.metric(
        "Holdout RMSE",
        f"£{backtest_rmse:,.0f}" if backtest_rmse is not None else "-",
        border=True,
    )
    st.metric(
        f"Next {horizon}-period forecast",
        f"£{float(future_mean.sum()):,.0f}",
        border=True,
    )

if naive_mape is not None and backtest_rmse is not None:
    if model_name == "Seasonal naive":
        st.caption(
            "This **is** the seasonal-naive baseline — the reference every other "
            "model's backtest is scored against."
        )
    else:
        skill = (1 - backtest_rmse / naive_rmse) * 100 if naive_rmse else float("nan")
        if np.isfinite(skill) and skill >= 0:
            verdict = f"**{model_name} beats** the seasonal-naive baseline by {skill:.0f}% on RMSE"
        else:
            verdict = f"**{model_name} trails** the seasonal-naive baseline by {abs(skill):.0f}% on RMSE"
        st.caption(
            f"Benchmark — seasonal-naive ('same period last year') holdout: "
            f"MAPE {naive_mape:.1f}%, RMSE £{naive_rmse:,.0f}. {verdict} (lower error is better)."
        )


# Main chart
st.space("small")
section("Forecast chart", eyebrow=f"{freq_label.lower()} revenue · {model_name}")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train.index, y=train.values,
    name="Training",
    mode="lines",
    line=dict(color="#3F6277", width=2, shape="spline"),
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
    name=f"{confidence_level}% CI (future)",
    showlegend=True,
))

fig.add_trace(go.Scatter(
    x=future_mean.index, y=future_mean.values,
    name="Forecast",
    mode="lines",
    line=dict(color="#B85F3D", width=2.5, shape="spline"),
))

historical_mean = float(series.mean())
fig.add_hline(
    y=historical_mean,
    line=dict(color=NEUTRAL_GRID, width=1, dash="dot"),
    annotation_text=f"Historical mean (£{historical_mean:,.0f})",
    annotation_position="top left",
    annotation_font_size=10,
    annotation_font_color="#6a6350",
)

fig.update_layout(
    title=f"{freq_label} Revenue - {model_name} Forecast",
    yaxis=dict(
        title="Revenue (£)",
        tickprefix="£",
        tickformat=",",
        rangemode="tozero",
    ),
    hovermode="x unified",
)
finalise_fig(fig, unified_hover=True)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Forecast mean and prediction band are clipped at £0 since revenue cannot be "
    "negative. If the lower band sits flush against the x-axis the underlying "
    "Gaussian interval extends below zero - read it as 'lower bound ≈ 0', not 'tight CI'."
)


# Residual diagnostics
st.space("small")
section("Diagnostics & forecast table", eyebrow="Residuals & values")
with st.expander(
    "Residual diagnostics",
    expanded=False,
    icon=":material/insights:",
):
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
        finalise_fig(fig_r, unified_hover=True)
        st.plotly_chart(fig_r, width="stretch")

    with d2:
        fig_h = go.Figure(go.Histogram(
            x=residuals.values,
            nbinsx=20,
            marker=dict(color="#B85F3D"),
        ))
        fig_h.update_layout(
            title="Residual distribution",
            xaxis_title="Residual (£)",
            yaxis_title="Count",
            showlegend=False,
        )
        finalise_fig(fig_h)
        st.plotly_chart(fig_h, width="stretch")

    st.caption(summary_caption)


# Forecast table
with st.expander(
    "Forecast values",
    expanded=False,
    icon=":material/table_chart:",
):
    lo_col = f"Lower {confidence_level}% (£)"
    hi_col = f"Upper {confidence_level}% (£)"
    fc_table = pd.DataFrame({
        "Period": future_mean.index.strftime("%Y-%m-%d"),
        "Forecast (£)": future_mean.values.astype(float),
        lo_col: future_lo.values.astype(float),
        hi_col: future_hi.values.astype(float),
    })
    st.dataframe(
        fc_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Forecast (£)": st.column_config.NumberColumn(format="£%.0f"),
            lo_col: st.column_config.NumberColumn(format="£%.0f"),
            hi_col: st.column_config.NumberColumn(format="£%.0f"),
        },
    )


st.caption(
    "Dataset covers ~2 years (Dec 2009 - Dec 2011). Yearly seasonality is "
    "identifiable but rests on only ~2 full cycles. Forecasts more than a "
    "season ahead should be treated as indicative."
)
