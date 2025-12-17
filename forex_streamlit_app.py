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
if 'analytics_tab' not in st.session_state:
    st.session_state.analytics_tab = 'overview'

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
            line=dict(color='#ff4b4b', width=2, dash='dash'),
            hovertemplate='%{y:.2f} (Predicted)<extra></extra>'
        ))

    fig.update_layout(
        title="", 
        xaxis_title="", 
        yaxis_title="",
        template="plotly_dark", 
        hovermode="x unified", 
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(size=10)
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
# CUSTOM UI STYLING - MOBILE OPTIMIZED
# ==========================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        max-width: 100%;
    }
    
    [data-testid="stMainBlockContainer"] {
        padding: 8px !important;
        max-width: 100%;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Card Styling */
    .signal-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 16px;
        border-radius: 16px;
        border: 1px solid #475569;
        margin-bottom: 12px;
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
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #475569;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: #334155;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        border-radius: 3px;
    }
    
    .advice-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        padding: 14px;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-top: 12px;
    }
    
    .tab-button {
        padding: 10px 16px;
        border: none;
        border-radius: 8px;
        background: #334155;
        color: #cbd5e1;
        font-weight: 600;
        cursor: pointer;
        font-size: 13px;
        margin: 4px 4px 4px 0;
        transition: all 0.3s;
        border: 1px solid #475569;
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-color: #10b981;
    }
    
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        color: white;
        transition: all 0.3s;
        font-size: 14px;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
    }
    
    .nav-buttons {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
        justify-content: space-between;
    }
    
    .nav-buttons button {
        flex: 1;
        font-size: 14px;
        font-weight: 600;
    }
    
    h1, h2, h3, h4, h5, h6 {
        margin: 12px 0 8px 0 !important;
    }
    
    h3 {
        font-size: 18px !important;
    }
    
    .divider {
        margin: 12px 0 !important;
    }
    
    .bottom-nav {
        display: flex;
        gap: 8px;
        padding: 12px 0;
        margin-top: 20px;
    }
    
    .bottom-nav button {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("### 🏆 LEADERBOARD")
    
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
            col1, col2, col3, col4 = st.columns([0.4, 1, 3, 1.2])
            with col1:
                st.markdown(f"<h3 style='text-align: center; margin: 0;'>{leader['rank']}</h3>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<h4 style='text-align: center; margin: 0;'>{leader['img']}</h4>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"**{leader['name']}**")
                if leader['badge']:
                    st.caption(f"🎖️ {leader['badge']}")
            with col4:
                st.markdown(f"<div style='text-align: right; color: #10b981; font-weight: 700;'>{leader['eff']}</div>", unsafe_allow_html=True)
            st.divider()

# ==========================================
# VIEW: MY SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.markdown("### 👛 MY SAVINGS")
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("<h1 style='text-align: center; font-size: 72px; margin: 0;'>☕</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 56px; line-height: 0.5; color: #10b981; margin: 12px 0;'>{st.session_state.coffee_count}</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>COFFEES SAVED</h4>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    savings_col1, savings_col2 = st.columns(2)
    with savings_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Savings</div>
            <div class="metric-value" style='font-size: 20px;'>₹3,420</div>
            <div style='color: #10b981; font-weight: 600; margin-top: 6px; font-size: 12px;'>↑ 12% this week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with savings_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Efficiency</div>
            <div class="metric-value" style='font-size: 20px;'>+2.4%</div>
            <div style='color: #94a3b8; margin-top: 6px; font-size: 12px;'>vs yesterday</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.button("📤 Share Savings", use_container_width=True)

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
            amt = st.number_input("Amount", value=1000, key="input_amt", label_visibility="collapsed", step=100)
        with c2:
            st.selectbox("Cur", ["INR"], label_visibility="collapsed", key="curr_from", disabled=True)
        
        converted = amt / current_rate
        st.markdown("<div style='text-align: center; color: #0066cc; font-size: 28px; margin: 12px 0;'>⬇️</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([3, 1])
        with c3:
            st.text_input("Converted", value=f"{converted:.2f}", disabled=True, label_visibility="collapsed")
        with c4:
            st.selectbox("Cur", ["EUR"], label_visibility="collapsed", key="curr_to", disabled=True)
        
        st.markdown("---")
        st.markdown("#### ⚡ Quick Picks")
        qp1, qp2, qp3, qp4 = st.columns(4)
        with qp1:
            if st.button("💲 USD", use_container_width=True): pass
        with qp2:
            if st.button("🇦🇺 AUD", use_container_width=True): pass
        with qp3:
            if st.button("🇨🇦 CAD", use_container_width=True): pass
        with qp4:
            if st.button("🇬🇧 GBP", use_container_width=True): pass
        
        if st.button("📝 LOG TRANSACTION", use_container_width=True):
            st.session_state.transaction_amt = amt
            st.session_state.trans_step = 'category'
            st.rerun()

    # --- UI Step 2: Categorize ---
    elif st.session_state.trans_step == 'category':
        st.markdown("### 🏷️ CATEGORIZE")
        cat_cols = st.columns(3)
        if cat_cols[0].button("🏠\nRent", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[1].button("🎓\nTuition", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[2].button("✈️\nTravel", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        cat_cols2 = st.columns(3)
        if cat_cols2[0].button("🛒\nShopping", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[1].button("👪\nRemittance", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[2].button("🔄\nOther", use_container_width=True): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        st.button("✅ CONFIRM", use_container_width=True)

    # --- ANALYTICS SECTION WITH TABS AND GRAPH ---
    else:
        st.divider()
        st.markdown("### 📊 MARKET ANALYSIS")
        
        # Run Models
        with st.spinner("🚀 Loading predictions..."):
            ols_model, ols_forecast = run_ols_model(df['Rate'], 30)
            arima_model, arima_forecast = run_arima_model(df['Rate'], 30)
            garch_model, garch_volatility = run_garch_model(df['Rate'], 30)
        
        # Calculate metrics
        confidence = calculate_confidence(ols_model)
        ols_dir = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
        final_pred = arima_forecast.iloc[-1]
        forecast_var = garch_model.forecast(horizon=30)
        risk_val = np.sqrt(forecast_var.variance.iloc[-1].values).mean()
        risk_level = "Low Risk" if risk_val < 0.5 else ("Medium Risk" if risk_val < 1.5 else "High Risk")
        daily_change = ((current_rate - df['Rate'].iloc[-2]) / df['Rate'].iloc[-2]) * 100
        
        # Top Signal Card
        signal_card_class = "signal-card positive" if "UP" in ols_dir else "signal-card negative"
        st.markdown(f"""
        <div class="{signal_card_class}">
            <div style='display: flex; justify-content: space-between;'>
                <div>
                    <div class='metric-label'>TODAY'S SIGNAL</div>
                    <div style='font-size: 28px; font-weight: 700; margin: 6px 0;'>{ols_dir}</div>
                    <div style='font-size: 12px; color: #cbd5e1;'>Confidence: <span style='color: #10b981;'>{confidence}%</span></div>
                </div>
                <div style='text-align: right;'>
                    <div class='metric-label'>RISK</div>
                    <div style='font-size: 16px; font-weight: 700; margin-top: 6px;'>{risk_level}</div>
                </div>
            </div>
            <div class='confidence-bar'>
                <div class='confidence-fill' style='width: {confidence}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # GRAPH - ALWAYS VISIBLE
        st.markdown("---")
        st.markdown("#### 📈 Market Graph")
        fig = create_interactive_plot(df, forecast_series=arima_forecast) 
        st.plotly_chart(fig, use_container_width=True)
        
        # Tab Navigation for Analytics Content
        st.markdown("---")
        tab_cols = st.columns(3)
        with tab_cols[0]:
            if st.button("📊 Overview", use_container_width=True, key="tab_overview"):
                st.session_state.analytics_tab = 'overview'
        with tab_cols[1]:
            if st.button("🎯 Forecast", use_container_width=True, key="tab_forecast"):
                st.session_state.analytics_tab = 'forecast'
        with tab_cols[2]:
            if st.button("💡 Details", use_container_width=True, key="tab_details"):
                st.session_state.analytics_tab = 'details'
        
        st.markdown("---")
        
        # TAB 1: OVERVIEW
        if st.session_state.analytics_tab == 'overview':
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Rate</div>
                    <div class="metric-value" style='font-size: 22px;'>₹{current_rate:.2f}</div>
                    <div style='color: {"#10b981" if daily_change > 0 else "#ef4444"}; font-weight: 600; margin-top: 6px; font-size: 12px;'>
                        {("↑" if daily_change > 0 else "↓")} {abs(daily_change):.3f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">30 Day Forecast</div>
                    <div class="metric-value" style='font-size: 22px;'>₹{final_pred:.2f}</div>
                    <div style='color: {"#10b981" if final_pred > current_rate else "#ef4444"}; font-weight: 600; margin-top: 6px; font-size: 12px;'>
                        {("↑ UP" if final_pred > current_rate else "↓ DOWN")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Key Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Volatility", f"{risk_val:.2f}%", delta=None)
            with metrics_col2:
                st.metric("Trend Strength", f"{confidence}%", delta=None)
        
        # TAB 2: FORECAST
        elif st.session_state.analytics_tab == 'forecast':
            st.markdown("#### 30-Day Forecast")
            
            forecast_df = pd.DataFrame({
                'Day': range(1, len(arima_forecast) + 1),
                'Predicted Rate': arima_forecast.values
            })
            
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # TAB 3: DETAILS
        elif st.session_state.analytics_tab == 'details':
            advice = generate_trading_advice(ols_dir, risk_level, current_rate, final_pred)
            st.markdown(f"""
            <div class="advice-box">
                <div style='font-size: 14px; font-weight: 700;'>🤖 AI RECOMMENDATION</div>
                <div style='font-size: 20px; margin-top: 8px; font-weight: 700;'>{advice}</div>
                <div style='margin-top: 8px; font-size: 12px; color: #cbd5e1;'>
                    Based on trend analysis and {30}-day prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Analysis Details")
            st.markdown(f"""
            - **Trend Direction**: {ols_dir}
            - **Risk Level**: {risk_level}
            - **Forecast Accuracy**: {confidence}%
            - **Current to Forecast Change**: ₹{abs(final_pred - current_rate):.2f} ({((final_pred - current_rate)/current_rate*100):.2f}%)
            """)

# ==========================================
# BOTTOM NAVIGATION - FIXED AT BOTTOM
# ==========================================
st.markdown("---")
st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
with nav_col1:
    if st.button("🔄 Convert", key="nav_convert", use_container_width=True): 
        st.session_state.nav = 'Convert'
        st.session_state.trans_step = 'input'
        st.rerun()
with nav_col2:
    if st.button("🏆 Leaderboard", key="nav_leader", use_container_width=True): 
        st.session_state.nav = 'Leaderboard'
        st.rerun()
with nav_col3:
    if st.button("👛 Savings", key="nav_savings", use_container_width=True): 
        st.session_state.nav = 'Savings'
        st.rerun()
