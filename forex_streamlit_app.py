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
    df_temp = pd.DataFrame({'Rate': series.values})
    df_temp['lag_1'] = df_temp['Rate'].shift(1)
    df_temp = df_temp.dropna()
    X = add_constant(df_temp[['lag_1']])
    y = df_temp['Rate']
    model = OLS(y, X).fit()
    forecast = [model.params[0] + model.params[1] * series.iloc[-1] for _ in range(forecast_steps)]
    return model, forecast

def run_arima_model(series, forecast_steps):
    model = ARIMA(series, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=forecast_steps).tolist()
    return model, forecast

def run_garch_model(series, forecast_steps):
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

# ==========================================
# MAIN APP
# ==========================================
st.markdown("### 💱 FOREX CONVERTER & SAVINGS")

# Always show market graph
col_graph1, col_graph2 = st.columns([3, 1])
with col_graph1:
    with st.spinner("📊 Fetching market data..."):
        df = fetch_data()
        ols_model, ols_forecast = run_ols_model(df['Rate'], 30)
        ols_direction = "UP" if ols_forecast[-1] > df['Rate'].iloc[-1] else "DOWN"
        ols_strength = ols_model.params[1]
        current_rate = df['Rate'].iloc[-1]
        fig = create_interactive_plot(df, ols_forecast[:7], "EUR/INR - 7 Day Forecast")
        st.plotly_chart(fig, use_container_width=True)

with col_graph2:
    returns = np.diff(np.log(df['Rate'].values)) * 100
    risk_val = np.std(returns)
    risk_level = "LOW" if risk_val < 0.5 else "MEDIUM" if risk_val < 1 else "HIGH"
    st.metric("Volatility Risk", risk_level, f"{risk_val:.2f}%")

# Navigation tabs
tab_graph, tab_overview, tab_forecast = st.tabs(["📊 Graph", "📈 Overview", "🔮 Forecast"])

with tab_graph:
    st.info("📈 Chart view is displayed above.")

with tab_overview:
    st.session_state.analytics_tab = 'Overview'
    with st.spinner("Running analysis models..."):
        arima_model, arima_forecast = run_arima_model(df['Rate'], 7)
        garch_fit, garch_volatility = run_garch_model(df['Rate'], 7)
        confidence = calculate_confidence(ols_model)
        advice, reason = generate_trading_advice(ols_direction, np.std(returns) / 100, current_rate, ols_forecast[-1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Signal</div>
                <div style='font-size: 32px; font-weight: 700; margin: 10px 0;'>{advice}</div>
                <div style='font-size: 12px; color: #cbd5e1;'>{reason}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div style='font-size: 28px; font-weight: 700; color: #3b82f6; margin: 10px 0;'>{confidence}%</div>
                <div style='width: 100%; background: #334155; height: 8px; border-radius: 4px; overflow: hidden;'>
                    <div style='width: {confidence}%; background: linear-gradient(90deg, #3b82f6, #10b981); height: 100%; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Rate</div>
                <div style='font-size: 28px; font-weight: 700; color: #fbbf24; margin: 10px 0;'>₹{current_rate:.2f}</div>
                <div style='font-size: 12px; color: #10b981;'>1 EUR = {current_rate:.2f} INR</div>
            </div>
            """, unsafe_allow_html=True)

with tab_overview:
    st.markdown("#### ARIMA Price Forecast")
    st.markdown(f"**7-day prediction:** {[f'₹{p:.2f}' for p in arima_forecast[:3]]}")
    st.markdown("#### GARCH Volatility")
    st.markdown(f"**Expected volatility:** {np.mean(garch_volatility):.4f}%")

# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("### 🏆 LEADERBOARD")
    
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "savings": "₹8,420", "img": "👩🏽", "badge": "MASTER", "streak": "15 days 🔥"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "savings": "₹6,850", "img": "👨🏽", "badge": "EXPERT", "streak": "12 days 🔥"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "savings": "₹5,230", "img": "👨🏻", "badge": "EXPERT", "streak": "8 days 🔥"},
        {"rank": "4", "name": "Rant K.", "eff": "+0.8%", "savings": "₹4,100", "img": "👩🏻", "badge": "PRO", "streak": "5 days"},
        {"rank": "5", "name": "Antre G.", "eff": "+0.7%", "savings": "₹3,560", "img": "👨🏾", "badge": "PRO", "streak": "3 days"},
        {"rank": "6", "name": "Divya K.", "eff": "+0.5%", "savings": "₹2,890", "img": "👩🏼", "badge": "", "streak": "2 days"},
    ]
    
    # User's Position
    st.markdown("#### 📍 Your Position")
    st.markdown(f"""
    <div class="metric-card">
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='text-align: left;'>
                <div class='metric-label'>Your Rank</div>
                <div style='font-size: 28px; font-weight: 700; color: #10b981;'>#12</div>
            </div>
            <div style='text-align: center;'>
                <div class='metric-label'>Weekly Savings</div>
                <div style='font-size: 24px; font-weight: 700;'>₹1,234</div>
            </div>
            <div style='text-align: right;'>
                <div class='metric-label'>Your Efficiency</div>
                <div style='font-size: 24px; font-weight: 700; color: #fbbf24;'>+0.3%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🏅 Top Savers This Week")
    
    for idx, leader in enumerate(leaders):
        card_style = "signal-card positive" if idx < 3 else "signal-card"
        medal_color = "#fbbf24" if idx == 0 else "#c0c0c0" if idx == 1 else "#cd7f32" if idx == 2 else "#cbd5e1"
        
        st.markdown(f"""
        <div class="{card_style}">
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; align-items: center; gap: 12px; flex: 1;'>
                    <div style='font-size: 32px; font-weight: 700; color: {medal_color}; min-width: 40px;'>{leader['rank']}</div>
                    <div style='font-size: 24px;'>{leader['img']}</div>
                    <div>
                        <div style='font-weight: 700; font-size: 16px; color: #ffffff;'>{leader['name']}</div>
                        <div style='font-size: 12px; color: #94a3b8;'>{leader['streak']}</div>
                    </div>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 14px; color: #94a3b8; text-transform: uppercase; font-weight: 600;'>Weekly</div>
                    <div style='font-size: 20px; font-weight: 700; color: #10b981; margin-bottom: 4px;'>{leader['savings']}</div>
                    <div style='font-size: 13px; color: #10b981; font-weight: 600;'>{leader['eff']}</div>
                </div>
            </div>
            <div style='display: flex; gap: 6px; margin-top: 10px;'>
                {f"<span style='background: #10b981; color: white; padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;'>🎖️ {leader['badge']}</span>" if leader['badge'] else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 💡 How to Climb the Leaderboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>📈</div>
            <div class='metric-label'>Track Trends</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Use market analysis to time transfers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>🎯</div>
            <div class='metric-label'>Build Streaks</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Daily activity multiplies savings</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 24px;'>💰</div>
            <div class='metric-label'>Save More</div>
            <div style='font-size: 12px; color: #cbd5e1; line-height: 1.4;'>Higher amounts = better rankings</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW: SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.markdown("### ☕ SAVINGS TRACKER")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Coffee Count</div>
            <div style='font-size: 48px; font-weight: 700; color: #f59e0b;'>{st.session_state.coffee_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Savings</div>
            <div style='font-size: 36px; font-weight: 700; color: #10b981;'>₹{st.session_state.coffee_count * 65:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Weekly Efficiency</div>
            <div style='font-size: 36px; font-weight: 700; color: #3b82f6;'>+0.3%</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW: CONVERT
# ==========================================
elif st.session_state.nav == 'Convert':
    if st.session_state.trans_step == 'input':
        st.markdown("### 💱 Convert Currency")
        amt = st.number_input("Amount (EUR)", min_value=1, value=100)
        st.session_state.transaction_amt = amt
        converted = amt * current_rate
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 20px; font-weight: 700;'>€ {amt} = ₹ {converted:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Next: Choose Category", key="next_cat"):
            st.session_state.trans_step = 'category'
            st.rerun()
    
    elif st.session_state.trans_step == 'category':
        st.markdown("### 📁 Select Category")
        cats = ["🏠 Living", "🍽️ Food", "✈️ Travel", "📚 Education", "🏥 Health", "💳 Other"]
        cols = st.columns(3)
        for i, cat in enumerate(cats):
            with cols[i % 3]:
                if st.button(cat, key=f"cat_{i}", use_container_width=True):
                    st.session_state.trans_step = 'analytics'
                    st.rerun()
    
    else:
        st.markdown("### 📊 Transaction Analytics")
        st.markdown(f"✅ Transaction recorded: €{st.session_state.transaction_amt} to INR")
        if st.button("New Transaction"):
            st.session_state.trans_step = 'input'
            st.rerun()

# ==========================================
# BOTTOM NAVIGATION
# ==========================================
st.markdown("---")
col_n1, col_n2, col_n3 = st.columns(3)
with col_n1:
    if st.button("🏆 Leaderboard", key="nav_lb", use_container_width=True):
        st.session_state.nav = 'Leaderboard'
        st.rerun()
with col_n2:
    if st.button("☕ Savings", key="nav_sv", use_container_width=True):
        st.session_state.nav = 'Savings'
        st.rerun()
with col_n3:
    if st.button("💱 Convert", key="nav_cv", use_container_width=True):
        st.session_state.nav = 'Convert'
        st.session_state.trans_step = 'input'
        st.rerun()
