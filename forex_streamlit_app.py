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
    """Creates an interactive plot with historical data and ARIMA forecast"""
    fig = go.Figure()

    # Historical Data
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Rate'], 
        mode='lines', 
        name='Historical Rate',
        line=dict(color='#00D9FF', width=3), 
        fill='tozeroy', 
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))

    # ARIMA Forecast
    if forecast_series is not None:
        last_hist_date = df.index[-1]
        last_hist_val = df['Rate'].iloc[-1]
        
        conn_x = [last_hist_date] + list(forecast_series.index)
        conn_y = [last_hist_val] + list(forecast_series.values)

        fig.add_trace(go.Scatter(
            x=conn_x,
            y=conn_y,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#FF6B9D', width=3, dash='dot'),
            hovertemplate='%{y:.2f} (Predicted)<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text="Rate History & Prediction", font=dict(size=18, color='#FFFFFF')),
        xaxis_title="", 
        yaxis_title="Rate (₹/EUR)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,40,0.5)',
        hovermode="x unified", 
        height=400, 
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='#FFFFFF')
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#FFFFFF'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#FFFFFF')
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

# ==========================================
# ENHANCED MOBILE-FIRST STYLING
# ==========================================
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Navigation buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 60px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    div.stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        padding: 15px !important;
        height: 70px !important;
    }
    
    /* Category buttons */
    .category-btn {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    /* Leaderboard styling */
    .leader-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Savings display */
    .savings-display {
        background: linear-gradient(135deg, #FF6B9D, #C471ED);
        border-radius: 30px;
        padding: 50px 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
    }
    
    /* Text styling */
    h1, h2, h3, p, div, span, label {
        color: white !important;
    }
    
    .big-number {
        font-size: 120px;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Quick picks */
    .quick-pick {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 20px 0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# NAVIGATION BAR WITH ICONS
# ==========================================
st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
col_n1, col_n2, col_n3 = st.columns(3)
with col_n1:
    if st.button("🔄\nConvert", key="nav_convert"): 
        st.session_state.nav = 'Convert'
        st.rerun()
with col_n2:
    if st.button("🏆\nLeaderboard", key="nav_leader"): 
        st.session_state.nav = 'Leaderboard'
        st.rerun()
with col_n3:
    if st.button("💰\nSavings", key="nav_savings"): 
        st.session_state.nav = 'Savings'
        st.rerun()

st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>🏆 Weekly Champions</h2>", unsafe_allow_html=True)
    
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "img": "👩🏽", "color": "linear-gradient(135deg, #FFD700, #FFA500)"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "img": "👨🏽", "color": "linear-gradient(135deg, #C0C0C0, #A8A8A8)"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "img": "👨🏻", "color": "linear-gradient(135deg, #CD7F32, #B8860B)"},
        {"rank": "4", "name": "Rant K.", "eff": "+0.8%", "img": "👩🏻", "color": "rgba(255, 255, 255, 0.15)"},
        {"rank": "5", "name": "Antre G.", "eff": "+0.7%", "img": "👨🏾", "color": "rgba(255, 255, 255, 0.12)"},
        {"rank": "6", "name": "Divya K.", "eff": "+0.5%", "img": "👩🏼", "color": "rgba(255, 255, 255, 0.1)"},
    ]
    
    for leader in leaders:
        st.markdown(f"""
        <div class='leader-card' style='background: {leader["color"]};'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='font-size: 32px;'>{leader['rank']}</span>
                    <span style='font-size: 36px;'>{leader['img']}</span>
                    <span style='font-size: 20px; font-weight: 600;'>{leader['name']}</span>
                </div>
                <div style='text-align: right;'>
                    <div style='font-size: 28px; font-weight: 700; color: #00FF88;'>{leader['eff']}</div>
                    <div style='font-size: 14px; opacity: 0.8;'>Efficiency</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# VIEW: MY SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
    
    col_center = st.columns([1, 6, 1])[1]
    with col_center:
        st.markdown(f"""
        <div class='savings-display'>
            <div style='font-size: 100px; margin-bottom: 20px;'>☕</div>
            <div class='big-number'>{st.session_state.coffee_count}</div>
            <div style='font-size: 32px; font-weight: 600; margin-top: 20px; letter-spacing: 3px;'>COFFEES SAVED</div>
            <div style='margin-top: 30px; font-size: 16px; opacity: 0.9;'>
                🎉 You're beating yesterday's rate!<br/>
                Keep up the smart trading!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        if st.button("📤 Share Achievement", type="primary"):
            st.balloons()
            st.success("Shared to your network! 🎊")

# ==========================================
# VIEW: CONVERT (Main Logic)
# ==========================================
elif st.session_state.nav == 'Convert':
    
    # Fetch Data for Rates
    df = fetch_data(period="1y")
    current_rate = df['Rate'].iloc[-1]
    
    # --- UI Step 1: Input ---
    if st.session_state.trans_step == 'input':
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>💱 Currency Converter</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            amt = st.number_input("", value=1000, key="input_amt", label_visibility="collapsed")
        with c2:
            st.selectbox("", ["INR"], label_visibility="collapsed", key="curr_from")
            
        st.markdown("<div style='text-align: center; font-size: 36px; margin: 10px 0;'>⬇️</div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns([4, 1])
        with c3:
            st.text_input("", value=f"{amt/current_rate:.2f}", disabled=True, label_visibility="collapsed")
        with c4:
            st.selectbox("", ["EUR"], label_visibility="collapsed", key="curr_to")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 14px; opacity: 0.7;'>Quick picks</p>", unsafe_allow_html=True)
        qp1, qp2, qp3, qp4 = st.columns(4)
        qp1.button("💲\nUSD")
        qp2.button("🇦🇺\nAUD")
        qp3.button("🇨🇦\nCAD")
        qp4.button("🇬🇧\nGBP")
        
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        if st.button("✨ LOG TRANSACTION", type="primary"):
            st.session_state.transaction_amt = amt
            st.session_state.trans_step = 'category'
            st.rerun()

    # --- UI Step 2: Categorize ---
    elif st.session_state.trans_step == 'category':
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>📊 Categorize Transaction</h2>", unsafe_allow_html=True)
        
        categories = [
            ("🏠", "Rent"),
            ("🎓", "Tuition"),
            ("✈️", "Travel"),
            ("🛒", "Shopping"),
            ("👪", "Family"),
            ("🔄", "Other")
        ]
        
        cat_cols = st.columns(3)
        for idx, (icon, label) in enumerate(categories):
            with cat_cols[idx % 3]:
                if st.button(f"{icon}\n{label}", key=f"cat_{idx}"): 
                    st.session_state.trans_step = 'input'
                    st.session_state.nav = 'Savings'
                    st.rerun()

    # --- Analytics Section ---
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    with st.expander("📊 AI Market Intelligence", expanded=True):
        
        forecast_days = st.slider("📅 Prediction Range (days)", 7, 90, 30)
        
        with st.spinner("🤖 Running AI models..."):
            ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
            arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
            garch_model, garch_volatility = run_garch_model(df['Rate'], forecast_days)

        # Interactive Chart
        fig = create_interactive_plot(df, forecast_series=arima_forecast) 
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics Row
        col_m1, col_m2, col_m3 = st.columns(3)
        
        ols_dir = "UP ↗️" if ols_model.params['Lag_1'] > 1 else "DOWN ↘️"
        final_pred = arima_forecast.iloc[-1]
        change_pct = ((final_pred - current_rate) / current_rate) * 100
        
        with col_m1:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 36px;'>📈</div>
                <div style='font-size: 24px; font-weight: 700; margin-top: 10px;'>{ols_dir}</div>
                <div style='font-size: 12px; opacity: 0.8;'>Trend</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 36px;'>🎯</div>
                <div style='font-size: 24px; font-weight: 700; margin-top: 10px;'>₹{final_pred:.2f}</div>
                <div style='font-size: 12px; opacity: 0.8;'>{forecast_days}d Forecast</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            color = "#00FF88" if change_pct > 0 else "#FF6B9D"
            st.markdown(f"""
            <div class='metric-box'>
                <div style='font-size: 36px;'>{'📊' if change_pct > 0 else '📉'}</div>
                <div style='font-size: 24px; font-weight: 700; margin-top: 10px; color: {color};'>{change_pct:+.2f}%</div>
                <div style='font-size: 12px; opacity: 0.8;'>Expected Change</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Trading Advice
        returns = df['Rate'].pct_change().dropna() * 100
        forecast_var = garch_model.fit(disp='off').forecast(horizon=forecast_days)
        risk_val = np.sqrt(forecast_var.variance.iloc[-1].values).mean()
        risk_level = "Low Risk" if risk_val < 0.5 else "High Risk"
        
        advice = generate_trading_advice(ols_dir, risk_level, current_rate, final_pred)
        
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card' style='text-align: center; background: linear-gradient(135deg, rgba(255, 107, 157, 0.2), rgba(196, 113, 237, 0.2));'>
            <div style='font-size: 18px; font-weight: 600; margin-bottom: 10px;'>🤖 AI Recommendation</div>
            <div style='font-size: 32px; font-weight: 800;'>{advice}</div>
        </div>
        """, unsafe_allow_html=True)
