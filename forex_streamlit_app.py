import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EUR/INR Forex Prediction",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💱 EUR/INR Exchange Rate Prediction")
st.subheader("Real-Time Forex Analysis with Econometric Models")

# ==========================================
# HELPER FUNCTIONS (from Notebook)
# ==========================================

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
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df['Rate'].iloc[-250:]
    ax1.plot(subset.index, subset, label='Historical (1 Year)', color='black', linewidth=2)
    
    last_date = df.index[-1]
    if ols_forecast and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = pd.date_range(start=last_date, periods=num_f+1, freq='D')[1:]
        dates_f = dates_f[dates_f.dayofweek < 5][:num_f]
        ax1.plot(dates_f, ols_forecast[:len(dates_f)], label='OLS Trend', linestyle='--', color='blue', alpha=0.7)
    
    if arima_forecast is not None and len(arima_forecast) > 0:
        num_f = len(arima_forecast)
        dates_f = pd.date_range(start=last_date, periods=num_f+1, freq='D')[1:]
        dates_f = dates_f[dates_f.dayofweek < 5][:num_f]
        ax1.plot(dates_f, arima_forecast.values[:len(dates_f)], label='ARIMA Forecast', linestyle='-', color='red', linewidth=2.5)
    
    ax1.axhline(y=current_rate, color='green', linestyle=':', linewidth=2, label='Current Rate', alpha=0.7)
    ax1.set_title('EUR/INR Exchange Rate Forecast', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rate (₹ per EUR)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df['MA_30'] = df['Rate'].rolling(window=30).mean()
    df['MA_90'] = df['Rate'].rolling(window=90).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2['Rate'], label='Daily Rate', color='black', alpha=0.4)
    ax2.plot(subset2.index, subset2['MA_30'], label='30-Day MA', color='blue', linewidth=2)
    ax2.plot(subset2.index, subset2['MA_90'], label='90-Day MA', color='red', linewidth=2)
    ax2.set_title('Trend Analysis (Moving Averages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate (₹ per EUR)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_trading_advice(ols_direction, risk, current, predicted):
    """Generate trading advice"""
    trend_up = "UP" in ols_direction
    risk_low = "Low" in risk
    price_up = predicted > current if predicted else False
    
    if trend_up and risk_low and price_up:
        return "✅ **STRONG BUY**: EUR appreciating with low volatility\n→ Optimal time to buy EUR with INR"
    elif trend_up and not risk_low and price_up:
        return "⚠️ **CAUTIOUS BUY**: EUR rising but high risk\n→ Use smaller positions with stop-losses"
    elif not trend_up and risk_low and not price_up:
        return "💡 **SELL SIGNAL**: EUR deprecating with low volatility\n→ Consider selling EUR for INR"
    elif not trend_up and not risk_low:
        return "🔴 **HIGH RISK DOWNTREND**: Avoid new positions\n→ Wait for market stabilization"
    else:
        return "🔄 **MIXED SIGNALS**: Unclear direction\n→ Monitor position closely"

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Settings")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y", "max"])

# ==========================================
# MAIN APP
# ==========================================

# Create two columns for metrics
col1, col2, col3 = st.columns(3)

# Load data
df = fetch_data(period=data_period)
current_rate = df['Rate'].iloc[-1]

with col1:
    st.metric("📊 Current Rate", f"₹{current_rate:.4f}", "EUR per 1 INR")

with col2:
    daily_change = ((df['Rate'].iloc[-1] - df['Rate'].iloc[-2]) / df['Rate'].iloc[-2]) * 100
    st.metric("📈 Daily Change", f"{daily_change:.3f}%", 
              delta=f"{daily_change:.3f}%")

with col3:
    st.metric("📅 Data Points", len(df), f"from {df.index[0].date()}")

st.divider()

# Run models
st.header("📊 Analysis Results")

col_analysis1, col_analysis2 = st.columns(2)

with st.spinner("🔄 Running econometric models..."):
    ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
    arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
    garch_model, garch_volatility = run_garch_model(df['Rate'], forecast_days)

# OLS Results
with col_analysis1:
    st.subheader("📈 OLS Trend Analysis")
    ols_direction = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
    ols_strength = round(ols_model.rsquared * 100, 1)
    
    st.info(f"**Direction:** {ols_direction}\n**Confidence:** {ols_strength}%")
    
    with st.expander("View OLS Model Summary"):
        st.write(ols_model.summary())

# ARIMA Results
with col_analysis2:
    st.subheader("🎯 ARIMA Forecast")
    if arima_forecast is not None and len(arima_forecast) > 0:
        final_pred = arima_forecast.iloc[-1]
        change_pct = ((final_pred - current_rate) / current_rate) * 100
        
        st.metric(f"Predicted Rate ({forecast_days}D)", f"₹{final_pred:.4f}",
                 delta=f"{change_pct:.3f}%")
        
        with st.expander("View ARIMA Model Summary"):
            st.write(arima_model.summary())

st.divider()

# GARCH Volatility
st.subheader("⚠️ Risk Assessment (GARCH Volatility)")

returns = df['Rate'].pct_change().dropna() * 100
garch_model_obj = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model_obj.fit(disp='off')
forecast_var = garch_fit.forecast(horizon=forecast_days)
avg_vol = np.sqrt(forecast_var.variance.iloc[-1].values).mean()

if avg_vol < 0.5:
    risk_level = "🟢 Low Risk"
    risk_desc = "Market is stable"
elif avg_vol < 1.0:
    risk_level = "🟡 Medium Risk"
    risk_desc = "Normal volatility expected"
else:
    risk_level = "🔴 High Risk"
    risk_desc = "Market is highly volatile"

st.info(f"**Volatility Index:** {avg_vol:.2f}\n**Risk Level:** {risk_level}\n{risk_desc}")

st.divider()

# Trading Advice
st.subheader("💡 Trading Recommendation")
advice = generate_trading_advice(ols_direction, risk_level, current_rate, 
                                final_pred if arima_forecast is not None else None)
st.warning(advice)

st.divider()

# Visualization
st.subheader("📉 Forecast Visualization")
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
