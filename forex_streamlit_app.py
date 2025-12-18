import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from models import (
    build_forecast_dates,
    classify_risk_from_variance,
    fetch_fx_data,
    generate_trading_advice,
    run_arima_model,
    run_garch_model,
    run_ols_model,
)
from transactions import append_transaction, load_transactions

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EUR/INR Forex Prediction",
    page_icon="💱",
    layout="centered",  # more comfortable on mobile
    initial_sidebar_state="expanded",
)

st.title("💱 EUR/INR Exchange Rate Prediction")
st.caption("Real-time forex analysis with econometric models – tuned for mobile screens.")

# ==========================================
# HELPER FUNCTIONS (from Notebook)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_data(period="1y"):
    """Fetch live EUR/INR data"""
    with st.spinner("📥 Loading EUR/INR data..."):
        ticker = "EURINR=X"
        data = yf.download(ticker, period=period, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = data[['Close']].copy()
        df = df.ffill()
        df['Rate'] = df['Close']
        df = df.resample('B').last().ffill()
        
        return df

def run_ols_model(series, forecast_steps):
    """Auto-Regressive OLS model"""
    df_ols = pd.DataFrame(series)
    df_ols['Lag_1'] = df_ols['Rate'].shift(1)
    df_ols.dropna(inplace=True)
    
    X = sm.add_constant(df_ols['Lag_1'])
    y = df_ols['Rate']
    
    model = sm.OLS(y, X).fit()
    
    last_val = series.iloc[-1]
    forecast = []
    
    for _ in range(forecast_steps):
        pred = model.params['const'] + model.params['Lag_1'] * last_val
        forecast.append(pred)
        last_val = pred
        
    return model, forecast

def run_arima_model(series, forecast_steps):
    """ARIMA model for forecasting"""
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    
    return model_fit, forecast_mean

def run_garch_model(series, forecast_steps):
    """GARCH model for volatility"""
    returns = 100 * series.pct_change().dropna()
    
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    
    forecast_res = model_fit.forecast(horizon=forecast_steps)
    variance_forecast = forecast_res.variance.iloc[-1].values
    
    return model_fit, variance_forecast

def create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate):
    """Create professional visualization"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df["Rate"].iloc[-180:]
    ax1.plot(subset.index, subset, label="Historical (≈6 Months)", color="black", linewidth=2)

    last_date = df.index[-1]
    if ols_forecast and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        ax1.plot(dates_f, ols_forecast[: len(dates_f)], label="OLS Trend", linestyle="--", color="blue", alpha=0.7)
    
    if arima_forecast is not None and len(arima_forecast) > 0:
        num_f = len(arima_forecast)
        dates_f = build_forecast_dates(last_date, num_f)
        ax1.plot(
            dates_f,
            arima_forecast.values[: len(dates_f)],
            label="ARIMA Forecast",
            linestyle="-",
            color="red",
            linewidth=2.5,
        )

    ax1.axhline(y=current_rate, color="green", linestyle=":", linewidth=2, label="Current Rate", alpha=0.7)
    ax1.set_title("EUR/INR Exchange Rate Forecast", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Rate (₹ per EUR)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df["MA_30"] = df["Rate"].rolling(window=30).mean()
    df["MA_90"] = df["Rate"].rolling(window=90).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2["Rate"], label="Daily Rate", color="black", alpha=0.4)
    ax2.plot(subset2.index, subset2["MA_30"], label="30-Day MA", color="blue", linewidth=2)
    ax2.plot(subset2.index, subset2["MA_90"], label="90-Day MA", color="red", linewidth=2)
    ax2.set_title("Trend Analysis (Moving Averages)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rate (₹ per EUR)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Settings")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y", "max"])

# ==========================================
# MAIN APP
# ==========================================

with st.spinner("📥 Loading EUR/INR data..."):
    df = fetch_fx_data(period=data_period)

if df.empty:
    st.error("No data available. Please try a different period.")
    st.stop()

current_rate = df["Rate"].iloc[-1]

st.subheader("Market Snapshot")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("📊 Current Rate (₹ / EUR)", f"{current_rate:.4f}")

with metric_col2:
    daily_change = ((df["Rate"].iloc[-1] - df["Rate"].iloc[-2]) / df["Rate"].iloc[-2]) * 100
    st.metric("📈 Daily Change", f"{daily_change:.3f}%", delta=f"{daily_change:.3f}%")

st.caption(f"Data points: **{len(df)}** (from {df.index[0].date()})")

# ==========================================
# TRANSACTION LOGGING (MOBILE-FRIENDLY)
# ==========================================
st.divider()
st.subheader("💾 Log Transaction")
st.caption("Track how many EUR you receive now vs. your last logged rate.")

tx_col_amount, tx_col_date = st.columns(2)

with tx_col_amount:
    amount_inr = st.number_input(
        "Amount in INR",
        min_value=0.0,
        step=1000.0,
        format="%.2f",
    )
with tx_col_date:
    tx_date = st.date_input("Transaction date")

log_button = st.button("Log Transaction", type="primary", use_container_width=True)

if log_button and amount_inr > 0:
    tx_df, savings_eur = append_transaction(tx_date, amount_inr, current_rate)
    eur_now = amount_inr / current_rate if current_rate > 0 else 0.0

    if savings_eur is None:
        st.success(
            f"Logged first transaction: **₹{amount_inr:,.2f} ➝ {eur_now:,.4f} EUR** "
            f"at rate **{current_rate:.4f} ₹/EUR**."
        )
    else:
        sign = "more" if savings_eur > 0 else "less"
        st.success(
            f"Logged transaction: **₹{amount_inr:,.2f} ➝ {eur_now:,.4f} EUR**.\n\n"
            f"Compared to your **previous logged rate**, you receive "
            f"**{abs(savings_eur):.4f} EUR {sign}** for the same INR amount."
        )

    with st.expander("View transaction history"):
        st.dataframe(tx_df.sort_values("date", ascending=False), use_container_width=True)

st.divider()

st.header("📊 Analysis Results")

with st.spinner("🔄 Running econometric models..."):
    ols_model, ols_forecast = run_ols_model(df["Rate"], forecast_days)
    arima_model, arima_forecast = run_arima_model(df["Rate"], forecast_days)
    garch_model, garch_variance = run_garch_model(df["Rate"], forecast_days)

ols_direction = "UP ↗" if ols_model.params["Lag_1"] > 1 else "DOWN ↘"
ols_strength = round(ols_model.rsquared * 100, 1)

st.subheader("📈 OLS Trend & 🎯 ARIMA Forecast")

ols_col, arima_col = st.columns(2)

with ols_col:
    st.markdown(f"**Direction:** {ols_direction}")
    st.markdown(f"**Confidence:** {ols_strength}%")
    with st.expander("View OLS Model Summary"):
        st.write(ols_model.summary())

with arima_col:
    if arima_forecast is not None and len(arima_forecast) > 0:
        final_pred = arima_forecast.iloc[-1]
        change_pct = ((final_pred - current_rate) / current_rate) * 100
        st.metric(f"Predicted Rate ({forecast_days}D)", f"₹{final_pred:.4f}", delta=f"{change_pct:.3f}%")
        with st.expander("View ARIMA Model Summary"):
            st.write(arima_model.summary())

st.divider()

# GARCH Volatility
st.subheader("⚠️ Risk Assessment (GARCH Volatility)")

risk_label, risk_desc, avg_vol = classify_risk_from_variance(garch_variance)
st.info(f"**Volatility Index:** {avg_vol:.2f}\n**Risk Level:** {risk_label}\n{risk_desc}")

st.divider()

# Trading Advice
st.subheader("💡 Trading Recommendation")
advice = generate_trading_advice(
    ols_direction,
    risk_label,
    current_rate,
    final_pred if ("final_pred" in locals() and arima_forecast is not None) else None,
)
st.warning(advice)

st.divider()

# Visualization
st.subheader("📉 Interactive Rate Chart (Last 6 Months)")
hist_df = df[["Rate"]].iloc[-180:].reset_index().rename(columns={"index": "Date"})
hist_df.rename(columns={hist_df.columns[0]: "Date"}, inplace=True)
st.line_chart(hist_df, x="Date", y="Rate", use_container_width=True)

st.subheader("📊 Forecast & Trend (Detailed)")
fig = create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate)
st.pyplot(fig)

st.divider()

# Historical Data Table
st.subheader("📋 Historical Data (Last 20 Days)")
st.dataframe(df[['Rate']].tail(20).style.format({"Rate": "{:.6f}"}))

# Footer
st.divider()
st.markdown("""
---
**About This App:**
- Uses **OLS** for trend analysis
- Uses **ARIMA(5,1,0)** for price forecasting
- Uses **GARCH(1,1)** for volatility assessment
- Data: EUR/INR exchange rates (₹ per 1 EUR)
- Developed for financial analysis and educational purposes
""")

