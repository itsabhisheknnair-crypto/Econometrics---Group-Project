import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="Forex Converter", page_icon="💱")

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] { background: #1e293b; }
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .metric-label { color: #94a3b8; font-size: 12px; font-weight: 500; text-transform: uppercase; margin-bottom: 6px; }
    .signal-card {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #64748b;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
    }
    .signal-card.positive { border-left-color: #10b981; }
    .signal-card.negative { border-left-color: #ef4444; }
    .signal-card.neutral { border-left-color: #f59e0b; }
    button { background: linear-gradient(135deg, #0f766e 0%, #115e59 100%); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: 600; min-height: 48px; }
    button:hover { background: linear-gradient(135deg, #0d5d5a 0%, #0f3f3a 100%); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'nav' not in st.session_state:
    st.session_state.nav = 'Convert'
if 'trans_step' not in st.session_state:
    st.session_state.trans_step = 'input'
if 'coffee_count' not in st.session_state:
    st.session_state.coffee_count = 19
if 'transaction_amt' not in st.session_state:
    st.session_state.transaction_amt = 0
if 'analytics_tab' not in st.session_state:
    st.session_state.analytics_tab = 'Overview'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fetch_data(period="1y"):
    df = yf.download("EURINR=X", period=period, progress=False)
    df = df.resample('B').last().dropna()
    return df[['Close']].rename(columns={'Close': 'Rate'})

def run_ols_model(series, forecast_steps):
    # Convert DataFrame to Series if needed
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df_temp = pd.DataFrame({'Rate': series.values})
    df_temp['lag_1'] = df_temp['Rate'].shift(1)
    df_temp = df_temp.dropna()
    X = add_constant(df_temp[['lag_1']])
    y = df_temp['Rate']
    model = OLS(y, X).fit()
    forecast = [model.params[0] + model.params[1] * series.iloc[-1] for _ in range(forecast_steps)]
    return model, forecast

def run_arima_model(series, forecast_steps):
    # Convert DataFrame to Series if needed
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    model = ARIMA(series, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=forecast_steps).tolist()
    return model, forecast

def run_garch_model(series, forecast_steps):
    # Convert DataFrame to Series if needed
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    returns = np.diff(np.log(series.values)) * 100
    garch_model_obj = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model_obj.fit(disp='off')
    forecast_var = garch_fit.forecast(horizon=forecast_steps)
    garch_volatility = np.sqrt(forecast_var.values[-1, :])
    return garch_fit, garch_volatility

def create_interactive_plot(df, forecast_series, title="EUR/INR Exchange Rate"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Rate'], mode='lines', name='Historical', line=dict(color='#10b981', width=2)))
    future_dates = pd.date_range(start=df.index[-1], periods=len(forecast_series)+1, freq='B')[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_series, mode='lines', name='Forecast', line=dict(color='#f59e0b', width=2, dash='dash')))
    fig.update_layout(template='plotly_dark', title=title, hovermode='x unified', height=350, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def generate_trading_advice(ols_direction, risk, current, predicted):
    if ols_direction == "UP" and risk < 0.05:
        return "BUY", "Signal suggests EUR appreciation. Favorable exchange rate."
    elif ols_direction == "DOWN" and risk < 0.05:
        return "SELL", "Signal suggests EUR depreciation. Lock in current rate."
    else:
        return "HOLD", "Mixed signals. Wait for clearer market direction."

def calculate_confidence(ols_model):
    return int(ols_model.rsquared * 100)
