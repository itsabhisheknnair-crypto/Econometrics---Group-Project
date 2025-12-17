import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import plotly.graph_objects as go  # Added for interactive graphs
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Easy Forex Predictor",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💱 EUR/INR Forex Explorer")
st.subheader("Learn how currency rates move and predict future prices")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def fetch_data(period="1y"):
    """Fetch live EUR/INR data"""
    with st.spinner("📥 Loading latest currency data..."):
        ticker = "EURINR=X"
        data = yf.download(ticker, period=period, progress=False)
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = data[['Close']].copy()
        df = df.ffill()
        df['Rate'] = df['Close']
        df = df.resample('B').last().ffill()
        
        return df

def run_ols_model(series, forecast_steps):
    """Simple Trend Line (OLS)"""
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
    """Prediction Model (ARIMA)"""
    # Simply using (1,1,1) for speed and stability in a demo app
    # In a real app, you might use auto_arima
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    
    return model_fit, forecast_mean

def run_garch_model(series, forecast_steps):
    """Risk/Volatility Model (GARCH)"""
    returns = 100 * series.pct_change().dropna()
    
    # Simple GARCH(1,1)
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    
    forecast_res = model_fit.forecast(horizon=forecast_steps)
    variance_forecast = forecast_res.variance.iloc[-1].values
    
    return model_fit, variance_forecast

def create_interactive_plot(df):
    """Create a student-friendly interactive plot using Plotly"""
    fig = go.Figure()

    # Add the main line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Rate'],
        mode='lines',
        name='Exchange Rate',
        line=dict(color='#0066cc', width=2),
        fill='tozeroy', # Fills area under the line slightly for better visuals
        fillcolor='rgba(0, 102, 204, 0.1)'
    ))

    fig.update_layout(
        title="Interactive Price History (Zoom & Hover)",
        xaxis_title="Time",
        yaxis_title="Rate (₹ per EUR)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate):
    """Create static visualization for reports"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Forecasts
    ax1 = axes[0]
    subset = df['Rate'].iloc[-250:]
    ax1.plot(subset.index, subset, label='History', color='black', linewidth=1.5)
    
    last_date = df.index[-1]
    if ols_forecast and len(ols_forecast) > 0:
        num_f = len(ols_forecast)
        dates_f = pd.date_range(start=last_date, periods=num_f+1, freq='D')[1:]
        dates_f = dates_f[dates_f.dayofweek < 5][:num_f]
        ax1.plot(dates_f, ols_forecast[:len(dates_f)], label='General Trend', linestyle='--', color='blue', alpha=0.7)
    
    if arima_forecast is not None and len(arima_forecast) > 0:
        num_f = len(arima_forecast)
        dates_f = pd.date_range(start=last_date, periods=num_f+1, freq='D')[1:]
        dates_f = dates_f[dates_f.dayofweek < 5][:num_f]
        ax1.plot(dates_f, arima_forecast.values[:len(dates_f)], label='AI Prediction', linestyle='-', color='red', linewidth=2.5)
    
    ax1.axhline(y=current_rate, color='green', linestyle=':', linewidth=2, label='Today Price', alpha=0.7)
    ax1.set_title('Future Prediction', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rate (₹)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Averages
    ax2 = axes[1]
    df['MA_30'] = df['Rate'].rolling(window=30).mean()
    subset2 = df.iloc[-250:]
    
    ax2.plot(subset2.index, subset2['Rate'], label='Daily Price', color='black', alpha=0.3)
    ax2.plot(subset2.index, subset2['MA_30'], label='30-Day Average', color='blue', linewidth=2)
    ax2.set_title('Average Price Trends', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate (₹)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_trading_advice(ols_direction, risk, current, predicted):
    """Generate simple advice"""
    trend_up = "UP" in ols_direction
    risk_low = "Low" in risk
    price_up = predicted > current if predicted else False
    
    if trend_up and risk_low and price_up:
        return "✅ **GOOD TIME TO BUY**: The Euro is growing stronger and the market is safe."
    elif trend_up and not risk_low and price_up:
        return "⚠️ **BE CAREFUL**: The Euro is rising, but the market is jumping around a lot."
    elif not trend_up and risk_low and not price_up:
        return "💡 **GOOD TIME TO SELL**: The Euro is losing value. Consider swapping back to Rupees."
    elif not trend_up and not risk_low:
        return "🔴 **STAY AWAY**: Prices are dropping and the market is crazy. Wait for it to calm down."
    else:
        return "🔄 **WAIT AND WATCH**: It's not clear where the price is going yet."

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.header("⚙️ Controls")
forecast_days = st.sidebar.slider("How far to predict? (Days)", 7, 90, 30)
data_period = st.sidebar.selectbox("History Length", ["1y", "2y", "5y", "max"])

# ==========================================
# MAIN APP
# ==========================================

# Load data
df = fetch_data(period=data_period)
current_rate = df['Rate'].iloc[-1]

# --- SECTION 1: CURRENCY CONVERTER (Like Screenshot) ---
st.divider()
st.subheader("💰 Quick Currency Converter")

# Using a container with a border to mimic the card look
with st.container(border=True):
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.caption("From (Euro)")
        # Number input for Euro
        amount = st.number_input("Amount", value=1000, label_visibility="collapsed", key="eur_input")
        st.write(f"🇪🇺 **EUR**")

    with col_c2:
        st.caption("To (Indian Rupee)")
        # Calculate conversion
        converted_value = amount * current_rate
        # Display large calculated number
        st.markdown(f"### ₹ {converted_value:,.2f}")
        st.write(f"🇮🇳 **INR**")
    
    # Exchange rate display
    st.caption(f"ℹ️ Exchange Rate: €1 = ₹{current_rate:.2f}")

st.divider()

# --- SECTION 2: INTERACTIVE GRAPH ---
st.subheader("📈 Interactive Rate History")
st.markdown("Use your mouse to **zoom in**, **pan**, and **hover** over specific dates to see exact prices.")

# Call the plotly function
interactive_fig = create_interactive_plot(df)
# Render using streamlit
st.plotly_chart(interactive_fig, use_container_width=True)

st.divider()

# --- SECTION 3: ANALYSIS ---
st.header("🔍 Market Analysis Results")

col_analysis1, col_analysis2 = st.columns(2)

with st.spinner("🔄 Calculating predictions..."):
    # Run the models
    ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
    arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
    garch_model, garch_volatility = run_garch_model(df['Rate'], forecast_days)

# OLS Results (Simplified text)
with col_analysis1:
    st.subheader("📊 General Trend")
    ols_direction = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
    ols_strength = round(ols_model.rsquared * 100, 1)
    
    st.info(f"**Direction:** {ols_direction}\n\nThis basically asks: *Is the price generally going up or down over time?*")

# ARIMA Results (Simplified text)
with col_analysis2:
    st.subheader("🎯 Future Prediction")
    if arima_forecast is not None and len(arima_forecast) > 0:
        final_pred = arima_forecast.iloc[-1]
        change_pct = ((final_pred - current_rate) / current_rate) * 100
        
        st.metric(f"Price in {forecast_days} Days", f"₹{final_pred:.4f}",
                 delta=f"{change_pct:.3f}%")
        st.caption("This uses a smart math model (AI) to guess the specific price in the future.")

st.divider()

# GARCH Volatility (Simplified text)
st.subheader("⚠️ Risk Level (Volatility)")

returns = df['Rate'].pct_change().dropna() * 100
garch_model_obj = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model_obj.fit(disp='off')
forecast_var = garch_fit.forecast(horizon=forecast_days)
avg_vol = np.sqrt(forecast_var.variance.iloc[-1].values).mean()

if avg_vol < 0.5:
    risk_level = "🟢 Low Risk"
    risk_desc = "The market is calm. Prices aren't jumping around much."
elif avg_vol < 1.0:
    risk_level = "🟡 Medium Risk"
    risk_desc = "Normal ups and downs. Be a little careful."
else:
    risk_level = "🔴 High Risk"
    risk_desc = "The market is crazy! Prices are swinging wildly."

st.info(f"**Risk Score:** {avg_vol:.2f} | **Status:** {risk_level}\n\n{risk_desc}")

st.divider()

# Trading Advice
st.subheader("💡 Simple Advice")
advice = generate_trading_advice(ols_direction, risk_level, current_rate, 
                                final_pred if arima_forecast is not None else None)
st.warning(advice)

st.divider()

# Visualization
st.subheader("📉 Prediction Charts (Static View)")
fig = create_visualization(df, forecast_days, ols_forecast, arima_forecast, current_rate)
st.pyplot(fig)

# Footer
st.divider()
st.markdown("""
---
**Student Guide:**
- **Trend:** Tells you if the line is going up or down.
- **Prediction:** A mathematical guess of where the price will be.
- **Risk:** How "bumpy" the ride will be. High risk means big sudden changes.
""")
