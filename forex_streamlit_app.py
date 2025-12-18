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
    elif trend_up and not risk_low and price_up: return "⚠ CAUTIOUS BUY"
    elif not trend_up and risk_low and not price_up: return "💡 SELL SIGNAL"
    else: return "🔄 HOLD/WAIT"

# ==========================================
# CUSTOM UI STYLING
# ==========================================
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-weight: bold;
    }
    .big-font { font-size: 20px !important; font-weight: bold; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# NAVIGATION BAR
# ==========================================================
# 1. Enhanced CSS for Mobile-First Bottom Navigation
st.markdown(
    """
    <style>
        /* Add padding to main content so it's not hidden behind nav */
        .main .block-container {
            padding-bottom: 100px !important;
        }
        
        /* Remove all default Streamlit styling that might interfere */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            width: 100% !important;
            background-color: #ffffff !important;
            padding: 8px 8px 8px 8px !important;
            z-index: 9999 !important;
            border-top: 1px solid #e0e0e0 !important;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.1) !important;
            margin: 0 !important;
        }
        
        /* Force the horizontal block to be flexbox and stay horizontal */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            justify-content: space-between !important;
            align-items: stretch !important;
            gap: 6px !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Make each column equal width and prevent stacking */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 !important;
            min-width: 0 !important;
            width: 33.333% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Ensure this works on mobile - override Streamlit's responsive behavior */
        @media only screen and (max-width: 768px) {
            section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) [data-testid="stHorizontalBlock"] {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
            }
            
            section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 !important;
                width: 33.333% !important;
                max-width: 33.333% !important;
            }
        }
        
        /* Style the buttons */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) button {
            width: 100% !important;
            height: 56px !important;
            border-radius: 10px !important;
            border: none !important;
            background-color: #f5f5f5 !important;
            color: #555 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 8px 4px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            box-sizing: border-box !important;
            cursor: pointer !important;
        }
        
        /* Button hover state */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) button:hover {
            background-color: #e8e8e8 !important;
        }
        
        /* Button active/pressed state */
        section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) button:active {
            background-color: #00D9A5 !important;
            color: white !important;
        }
        
        /* Mobile specific button sizing */
        @media only screen and (max-width: 480px) {
            section[data-testid="stVerticalBlock"] > div:has(div.sticky-nav) button {
                font-size: 11px !important;
                height: 52px !important;
                padding: 6px 2px !important;
            }
        }
        
        /* Hide the sticky-nav marker div itself */
        .sticky-nav {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Navigation State Logic
if 'nav' not in st.session_state:
    st.session_state.nav = 'Convert'

# 3. Page Content (Based on selected navigation)
if st.session_state.nav == 'Convert':
    st.header("🔄 Convert Currency")
    
    st.markdown("**Market Trend:** DOWN ↘")
    st.markdown("**Forecast (30 days):** ₹106.43")
    
    # Add scrollable content
    for i in range(30):
        st.write(f"Content line {i+1}")

elif st.session_state.nav == 'Leaderboard':
    st.header("🏆 Leaderboard")
    st.write("Current top players:")
    st.table({
        "Rank": ["1", "2", "3"],
        "User": ["Alice", "Bob", "Charlie"], 
        "Score": [1000, 850, 720]
    })

elif st.session_state.nav == 'Savings':
    st.header("👛 My Savings")
    st.metric("Total Balance", "$4,250.00")
    st.write("Your savings are growing!")

# 4. Bottom Navigation Bar (Must be at the end)
with st.container():
    st.markdown('<div class="sticky-nav"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Convert", key="btn_convert", use_container_width=True):
            st.session_state.nav = 'Convert'
            st.rerun()
    
    with col2:
        if st.button("🏆 Leaderboard", key="btn_leader", use_container_width=True):
            st.session_state.nav = 'Leaderboard'
            st.rerun()
    
    with col3:
        if st.button("👛 My Savings", key="btn_savings", use_container_width=True):
            st.session_state.nav = 'Savings'
            st.rerun()
# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.subheader("WEEKLY SAVINGS LEADERBOARD")
    t1, t2, t3 = st.columns([1,2,1])
    with t2:
        st.caption("Friends 🔘 Global")
    
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "img": "👩🏽"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "img": "👨🏽"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "img": "👨🏻"},
        {"rank": "2", "name": "Rant K.", "eff": "+0.8%", "img": "👩🏻"},
        {"rank": "4", "name": "Antre G.", "eff": "+0.7%", "img": "👨🏾"},
        {"rank": "5", "name": "Divya K.", "eff": "+0.5%", "img": "👩🏼"},
    ]
    
    for leader in leaders:
        with st.container():
            c1, c2, c3 = st.columns([1, 4, 2])
            with c1: st.write(f"### {leader['rank']}")
            with c2: st.write(f"{leader['img']} {leader['name']}")
            with c3: st.success(f"{leader['eff']}\nEfficiency")
            st.markdown("---")

# ==========================================
# VIEW: MY SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.subheader("MY SAVINGS WALLET")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>☕</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; font-size: 100px; line-height: 0.5;'>{st.session_state.coffee_count}</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>COFFEES</h3>", unsafe_allow_html=True)
        st.info(f"You saved *{st.session_state.coffee_count} COFFEES* from your last transaction!\n\nCompared to yesterday's rate.")
        st.button("Share my Savings", type="primary")

# ==========================================
# VIEW: CONVERT (Main Logic)
# ==========================================
elif st.session_state.nav == 'Convert':
    
    # Fetch Data for Rates
    df = fetch_data(period="1y")
    current_rate = df['Rate'].iloc[-1]
    
    # --- UI Step 1: Input ---
    if st.session_state.trans_step == 'input':
        st.subheader("Log Transaction")
        c1, c2 = st.columns([3, 1])
        with c1:
            amt = st.number_input("Amount", value=1000, key="input_amt", label_visibility="collapsed")
        with c2:
            st.selectbox("Cur", ["INR"], label_visibility="collapsed", key="curr_from")
            
        st.markdown("<div style='text-align: center; color: #0066cc; font-size: 24px;'>⬇</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([3, 1])
        with c3:
            st.text_input("Converted", value=f"{amt/current_rate:.2f}", disabled=True, label_visibility="collapsed")
        with c4:
            st.selectbox("Cur", ["EUR"], label_visibility="collapsed", key="curr_to")
            
        st.caption("Quick picks")
        qp1, qp2, qp3, qp4 = st.columns(4)
        qp1.button("💲 USD")
        qp2.button("🇦🇺 AUD")
        qp3.button("🇨🇦 CAD")
        qp4.button("🇬🇧 GBP")
        
        if st.button("LOG TRANSACTION", type="primary"):
            st.session_state.transaction_amt = amt
            st.session_state.trans_step = 'category'
            st.rerun()

    # --- UI Step 2: Categorize ---
    elif st.session_state.trans_step == 'category':
        st.subheader("CATEGORIZE YOUR TRANSACTION")
        cat_cols = st.columns(3)
        if cat_cols[0].button("🏠\nRent"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[1].button("🎓\nTuition"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols[2].button("✈\nTravel"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        cat_cols2 = st.columns(3)
        if cat_cols2[0].button("🛒\nShopping"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[1].button("👪\nRemittance"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if cat_cols2[2].button("🔄\nOther"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        st.button("CONFIRM & SEE SAVINGS", type="primary")

    # --- BELOW: Analytics with UPDATED Graph ---
    st.divider()
    with st.expander("📊 Advanced Market Analysis & Prediction", expanded=True):
        
        # 1. Controls
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        
        # 2. Run Models FIRST (so we have data for the plot)
        with st.spinner("Running AI Prediction Models..."):
            ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
            arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
            garch_model, garch_volatility = run_garch_model(df['Rate'], forecast_days)

        # 3. Interactive Plot (Now includes arima_forecast)
        st.subheader("Market Graph")
        # Pass the ARIMA forecast to the plotting function here
        fig = create_interactive_plot(df, forecast_series=arima_forecast) 
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Text Analysis
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.subheader("Trend Analysis")
            ols_dir = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
            st.info(f"Market Trend: *{ols_dir}*")
            
        with col_res2:
            st.subheader("Price Prediction")
            final_pred = arima_forecast.iloc[-1]
            st.success(f"Forecast ({forecast_days} days): *₹{final_pred:.2f}*")
            
        # 5. Advice
        returns = df['Rate'].pct_change().dropna() * 100
        forecast_var = garch_model.fit(disp='off').forecast(horizon=forecast_days)
        risk_val = np.sqrt(forecast_var.variance.iloc[-1].values).mean()
        risk_level = "Low Risk" if risk_val < 0.5 else "High Risk"
        
        advice = generate_trading_advice(ols_dir, risk_level, current_rate, final_pred)
        st.warning(f"AI Recommendation: *{advice}*")











