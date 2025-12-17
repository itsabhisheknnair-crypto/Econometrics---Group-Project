import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Forex Savings & Predictor",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
if 'nav' not in st.session_state:
    st.session_state.nav = 'Convert'
if 'trans_step' not in st.session_state:
    st.session_state.trans_step = 'input'
if 'coffee_count' not in st.session_state:
    st.session_state.coffee_count = 19
if 'transaction_amt' not in st.session_state:
    st.session_state.transaction_amt = 0

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fetch_data(period="1y"):
    """Fetch live EUR/INR data"""
    with st.spinner("📥 Loading latest currency data..."):
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
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    
    # Ensure index is datetime for plotting if it isn't already
    if not isinstance(forecast_mean.index, pd.DatetimeIndex):
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='B')[1:]
        forecast_mean.index = forecast_dates
        
    return model_fit, forecast_mean

def run_garch_model(series, forecast_steps):
    returns = 100 * series.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast_res = model_fit.forecast(horizon=forecast_steps)
    variance_forecast = forecast_res.variance.iloc[-1].values
    return model_fit, variance_forecast

def create_interactive_plot(df, forecast_series=None):
    """
    Creates an interactive plot with historical data and optionally the ARIMA forecast.
    """
    fig = go.Figure()

    # 1. Historical Data Trace
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Rate'], 
        mode='lines', 
        name='Historical Rate',
        line=dict(color='#0066cc', width=2), 
        fill='tozeroy', 
        fillcolor='rgba(0, 102, 204, 0.1)'
    ))

    # 2. ARIMA Forecast Trace (if provided)
    if forecast_series is not None:
        # We add the last historical point to the forecast series to make the lines connect visually
        last_hist_date = df.index[-1]
        last_hist_val = df['Rate'].iloc[-1]
        
        # Create a connecting series
        conn_x = [last_hist_date] + list(forecast_series.index)
        conn_y = [last_hist_val] + list(forecast_series.values)

        fig.add_trace(go.Scatter(
            x=conn_x,
            y=conn_y,
            mode='lines',
            name='ARIMA Prediction',
            line=dict(color='#ff4b4b', width=2, dash='dash'), # Red dashed line for prediction
            hovertemplate='%{y:.2f} (Predicted)<extra></extra>'
        ))

    fig.update_layout(
        title="Interactive Price History & Prediction", 
        xaxis_title="Time", 
        yaxis_title="Rate (₹ per EUR)",
        template="plotly_white", 
        hovermode="x unified", 
        height=450, 
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def generate_trading_advice(ols_direction, risk, current, predicted):
    trend_up = "UP" in ols_direction
    risk_low = "Low" in risk
    price_up = predicted > current if predicted else False
    if trend_up and risk_low and price_up: return "✅ BUY SIGNAL"
    elif trend_up and not risk_low and price_up: return "⚠️ CAUTIOUS BUY"
    elif not trend_up and risk_low and not price_up: return "💡 SELL SIGNAL"
    else: return "🔄 HOLD/WAIT"

def calculate_confidence(ols_model):
    """Calculate confidence score based on model R-squared"""
    r_squared = ols_model.rsquared
    confidence = min(int(r_squared * 100), 100)
    return confidence

# ==========================================
# CUSTOM UI STYLING
# ==========================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
    }
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Card Styling */
    .signal-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #475569;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .signal-card.positive {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        border-color: #52b788;
    }
    
    .signal-card.negative {
        background: linear-gradient(135deg, #4a1f1f 0%, #6a2c2c 100%);
        border-color: #d62828;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #475569;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .confidence-bar {
        width: 100%;
        height: 8px;
        background: #334155;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        border-radius: 4px;
    }
    
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 24px 0 16px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .advice-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-top: 16px;
    }
    
    .risk-low {
        color: #10b981;
    }
    
    .risk-medium {
        color: #f59e0b;
    }
    
    .risk-high {
        color: #ef4444;
    }
    
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        color: white;
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
    }
    
    .nav-button {
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        border: 2px solid #475569;
        background: transparent;
        color: #e2e8f0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-color: #10b981;
        color: white;
    }
    
    .savings-highlight {
        font-size: 24px;
        color: #10b981;
        font-weight: 700;
    }
    
    .price-arrow-up {
        color: #10b981;
        font-size: 24px;
    }
    
    .price-arrow-down {
        color: #ef4444;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# NAVIGATION BAR
# ==========================================
col_n1, col_n2, col_n3 = st.columns(3)
with col_n1:
    if st.button("🔄 Convert", key="nav_convert"): 
        st.session_state.nav = 'Convert'
        st.rerun()
with col_n2:
    if st.button("🏆 Leaderboard", key="nav_leader"): 
        st.session_state.nav = 'Leaderboard'
        st.rerun()
with col_n3:
    if st.button("👛 My Savings", key="nav_savings"): 
        st.session_state.nav = 'Savings'
        st.rerun()

st.divider()

# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("### 🏆 WEEKLY SAVINGS LEADERBOARD")
    
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "img": "👩🏽", "badge": "MASTER"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "img": "👨🏽", "badge": "EXPERT"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "img": "👨🏻", "badge": "EXPERT"},
        {"rank": "4", "name": "Rant K.", "eff": "+0.8%", "img": "👩🏻", "badge": "PRO"},
        {"rank": "5", "name": "Antre G.", "eff": "+0.7%", "img": "👨🏾", "badge": "PRO"},
        {"rank": "6", "name": "Divya K.", "eff": "+0.5%", "img": "👩🏼", "badge": ""},
    ]
    
    for leader in leaders:
        with st.container():
            col1, col2, col3, col4 = st.columns([0.5, 1, 3, 1.5])
            with col1:
                st.markdown(f"<h2 style='text-align: center;'>{leader['rank']}</h2>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>{leader['img']}</h3>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"**{leader['name']}**")
                if leader['badge']:
                    st.caption(f"🎖️ {leader['badge']}")
            with col4:
                st.markdown(f"<div style='text-align: right; color: #10b981; font-weight: 700; font-size: 18px;'>{leader['eff']}</div>", unsafe_allow_html=True)
            st.divider()

# ==========================================
# VIEW: MY SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.markdown("### 👛 MY SAVINGS WALLET")
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>☕</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 100px; line-height: 0.5; color: #10b981;'>{st.session_state.coffee_count}</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>COFFEES SAVED</h3>", unsafe_allow_html=True)
    
    savings_col1, savings_col2 = st.columns(2)
    with savings_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Savings</div>
            <div class="metric-value">₹3,420</div>
            <div style='color: #10b981; font-weight: 600; margin-top: 8px;'>↑ 12% this week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with savings_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Efficiency Rate</div>
            <div class="metric-value">+2.4%</div>
            <div style='color: #94a3b8; margin-top: 8px;'>vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.button("Share my Savings 📤", type="primary")

# ==========================================
# VIEW: CONVERT (Main Logic)
# ==========================================
elif st.session_state.nav == 'Convert':
    
    # Fetch Data for Rates
    df = fetch_data(period="1y")
    current_rate = df['Rate'].iloc[-1]
    
    # --- UI Step 1: Input ---
    if st.session_state.trans_step == 'input':
        st.markdown("### 📊 TODAY'S SIGNAL")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            amt = st.number_input("Amount", value=1000, key="input_amt", label_visibility="collapsed")
        with c2:
            st.selectbox("Cur", ["INR"], label_visibility="collapsed", key="curr_from")
        
        converted = amt / current_rate
        st.markdown("<div style='text-align: center; color: #0066cc; font-size: 32px; margin: 16px 0;'>⬇️</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([3, 1])
        with c3:
            st.text_input("Converted", value=f"{converted:.2f}", disabled=True, label_visibility="collapsed")
        with c4:
            st.selectbox("Cur", ["EUR"], label_visibility="collapsed", key="curr_to")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Picks")
        qp1, qp2, qp3, qp4 = st.columns(4)
        with qp1:
            st.button("💲 USD")
        with qp2:
            st.button("🇦🇺 AUD")
        with qp3:
            st.button("🇨🇦 CAD")
        with qp4:
            st.button("🇬🇧 GBP")
        
        if st.button("LOG TRANSACTION", type="primary", use_container_width=True):
            st.session_state.transaction_amt = amt
            st.session_state.trans_step = 'category'
            st.rerun()

    # --- UI Step 2: Categorize ---
    elif st.session_state.trans_step == 'category':
        st.markdown("### 🏷️ CATEGORIZE YOUR TRANSACTION")
        cat_cols = st.columns(3)
        if cat_cols[0].button("🏠 Rent"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[1].button("🎓 Tuition"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[2].button("✈️ Travel"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        cat_cols2 = st.columns(3)
        if cat_cols2[0].button("🛒 Shopping"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[1].button("👪 Remittance"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[2].button("🔄 Other"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        st.button("CONFIRM & SEE SAVINGS", type="primary", use_container_width=True)

    # --- BELOW: Analytics with UPDATED Graph ---
    st.divider()
    
    st.markdown("### 📊 ADVANCED MARKET ANALYSIS & PREDICTION")
    
    # 1. Controls
    forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    # 2. Run Models FIRST (so we have data for the plot)
    with st.spinner("🚀 Running AI Prediction Models..."):
        ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
        arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
        garch_model, garch_volatility = run_garch_model(df['Rate'], forecast_days)
    
    # Calculate metrics
    confidence = calculate_confidence(ols_model)
    ols_dir = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
    final_pred = arima_forecast.iloc[-1]
    returns = df['Rate'].pct_change().dropna() * 100
    forecast_var = garch_model.forecast(horizon=forecast_days)
    risk_val = np.sqrt(forecast_var.variance.iloc[-1].values).mean()
    risk_level = "Low Risk" if risk_val < 0.5 else ("Medium Risk" if risk_val < 1.5 else "High Risk")
    daily_change = ((current_rate - df['Rate'].iloc[-2]) / df['Rate'].iloc[-2]) * 100
    savings = st.session_state.transaction_amt * (abs(final_pred - current_rate) / current_rate)
    
    # 3. Top Signal Card
    signal_card_class = "signal-card positive" if "UP" in ols_dir else "signal-card negative"
    st.markdown(f"""
    <div class="{signal_card_class}">
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <div class='metric-label'>TODAY'S SIGNAL</div>
                <div style='font-size: 32px; font-weight: 700; margin: 8px 0;'>{ols_dir}</div>
                <div style='font-size: 14px; color: #cbd5e1;'>Confidence: <span style='color: #10b981; font-weight: 700;'>{confidence}%</span></div>
            </div>
            <div style='text-align: right;'>
                <div class='metric-label'>RISK LEVEL</div>
                <div style='font-size: 20px; font-weight: 700; margin-top: 8px;'>{risk_level}</div>
            </div>
        </div>
        <div class='confidence-bar'>
            <div class='confidence-fill' style='width: {confidence}%;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Interactive Plot
    st.markdown("#### 📈 Market Graph")
    fig = create_interactive_plot(df, forecast_series=arima_forecast) 
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Metrics Grid
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Rate</div>
            <div class="metric-value">₹{current_rate:.2f}</div>
            <div style='color: {"#10b981" if daily_change > 0 else "#ef4444"}; font-weight: 600; margin-top: 8px;'>
                {("↑" if daily_change > 0 else "↓")} {abs(daily_change):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{forecast_days} Day Forecast</div>
            <div class="metric-value">₹{final_pred:.2f}</div>
            <div style='color: {"#10b981" if final_pred > current_rate else "#ef4444"}; font-weight: 600; margin-top: 8px;'>
                {("↑ UP" if final_pred > current_rate else "↓ DOWN")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Potential Savings</div>
            <div class="metric-value" style='color: #10b981;'>₹{savings:.0f}</div>
            <div style='color: #94a3b8; margin-top: 8px;'>from ₹{st.session_state.transaction_amt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 6. AI Recommendation
    advice = generate_trading_advice(ols_dir, risk_level, current_rate, final_pred)
    st.markdown(f"""
    <div class="advice-box">
        <div style='font-size: 16px; font-weight: 700;'>🤖 AI RECOMMENDATION</div>
        <div style='font-size: 24px; margin-top: 12px; font-weight: 700;'>{advice}</div>
        <div style='margin-top: 12px; font-size: 14px; color: #cbd5e1;'>
            Based on trend analysis, risk assessment, and price predictions over {forecast_days} days.
        </div>
    </div>
    """, unsafe_allow_html=True)
