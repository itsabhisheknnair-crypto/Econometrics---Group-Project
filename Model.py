import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def fetch_fx_data(ticker: str = "EURINR=X", period: str = "1y") -> pd.DataFrame:
    """Download and preprocess FX data from Yahoo Finance."""
    data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        return pd.DataFrame()

    # Normalize columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Close"]].copy()
    df = df.ffill()
    df.rename(columns={"Close": "Rate"}, inplace=True)

    # Resample to business days
    df = df.resample("B").last().ffill()
    return df


def run_ols_model(series: pd.Series, forecast_steps: int):
    """Auto-Regressive OLS model with 1 lag."""
    df_ols = pd.DataFrame({"Rate": series}).copy()
    df_ols["Lag_1"] = df_ols["Rate"].shift(1)
    df_ols.dropna(inplace=True)

    X = sm.add_constant(df_ols["Lag_1"])
    y = df_ols["Rate"]

    model = sm.OLS(y, X).fit()

    last_val = series.iloc[-1]
    forecast = []

    for _ in range(forecast_steps):
        pred = model.params["const"] + model.params["Lag_1"] * last_val
        forecast.append(pred)
        last_val = pred

    return model, forecast


def run_arima_model(series: pd.Series, forecast_steps: int):
    """ARIMA model for forecasting."""
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean

    return model_fit, forecast_mean


def run_garch_model(series: pd.Series, forecast_steps: int):
    """GARCH(1,1) model for volatility forecasting."""
    returns = 100 * series.pct_change().dropna()

    model = arch_model(returns, vol="Garch", p=1, q=1)
    model_fit = model.fit(disp="off")

    forecast_res = model_fit.forecast(horizon=forecast_steps)
    variance_forecast = forecast_res.variance.iloc[-1].values

    return model_fit, variance_forecast


def classify_risk_from_variance(variance_forecast: np.ndarray) -> tuple[str, str, float]:
    """Summarize GARCH variance forecast into a human-readable risk bucket."""
    avg_vol = np.sqrt(variance_forecast).mean()

    if avg_vol < 0.5:
        return "🟢 Low Risk", "Market is stable", avg_vol
    elif avg_vol < 1.0:
        return "🟡 Medium Risk", "Normal volatility expected", avg_vol
    else:
        return "🔴 High Risk", "Market is highly volatile", avg_vol


def generate_trading_advice(ols_direction: str, risk_label: str, current: float, predicted: float | None) -> str:
    """Generate trading advice based on trend, risk and relative price."""
    trend_up = "UP" in ols_direction
    risk_low = "Low" in risk_label
    price_up = predicted > current if predicted is not None else False

    if trend_up and risk_low and price_up:
        return "✅ **STRONG BUY**: EUR appreciating with low volatility\n→ Optimal time to buy EUR with INR"
    elif trend_up and not risk_low and price_up:
        return "⚠️ **CAUTIOUS BUY**: EUR rising but high risk\n→ Use smaller positions with stop-losses"
    elif not trend_up and risk_low and not price_up:
        return "💡 **SELL SIGNAL**: EUR depreciating with low volatility\n→ Consider selling EUR for INR"
    elif not trend_up and not risk_low:
        return "🔴 **HIGH RISK DOWNTREND**: Avoid new positions\n→ Wait for market stabilization"
    else:
        return "🔄 **MIXED SIGNALS**: Unclear direction\n→ Monitor position closely"


def build_forecast_dates(last_index: pd.DatetimeIndex | pd.Timestamp, steps: int):
    """Utility to build business-day forecast dates."""
    last_date = last_index if isinstance(last_index, pd.Timestamp) else last_index[-1]
    dates_f = pd.date_range(start=last_date, periods=steps + 1, freq="D")[1:]
    dates_f = dates_f[dates_f.dayofweek < 5][:steps]
    return dates_f


