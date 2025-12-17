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
    layout="mobile", # Changed to mobile layout to match the app-like screenshots
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
    # Using caching to speed up reruns and prevent constant reloading
    @st.cache_data(ttl=3600)
    def load_ticker_data():
        ticker = "EURINR=X"
        data = yf.download(ticker, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data[['Close']].copy()
        df = df.ffill()
        df['Rate'] = df['Close']
        df = df.resample('B').last().ffill()
        return df
    
    return load_ticker_data()

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
    # Return the fit result and the variance array directly
    variance_forecast = forecast_res.variance.iloc[-1].values
    return model_fit, variance_forecast

def create_interactive_plot(df, forecast_series=None):
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Rate'], mode='lines', name='Historical Rate',
        line=dict(color='#0066cc', width=2), fill='tozeroy', fillcolor='rgba(0, 102, 204, 0.1)'
    ))

    # ARIMA Forecast
    if forecast_series is not None:
        last_hist_date = df.index[-1]
        last_hist_val = df['Rate'].iloc[-1]
        conn_x = [last_hist_date] + list(forecast_series.index)
        conn_y = [last_hist_val] + list(forecast_series.values)

        fig.add_trace(go.Scatter(
            x=conn_x, y=conn_y, mode='lines', name='ARIMA Prediction',
            line=dict(color='#ff4b4b', width=2, dash='dash'),
            hovertemplate='%{y:.2f} (Predicted)<extra></extra>'
        ))

    fig.update_layout(
        title="Interactive Price History & Prediction",
        xaxis_title="Time", yaxis_title="Rate (₹ per EUR)",
        template="plotly_white", hovermode="x unified",
        height=400, margin=dict(l=20, r=20, t=50, b=20),
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

# ==========================================
# CUSTOM UI STYLING
# ==========================================
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        border-radius: 25px;
        height: 50px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    .big-font { font-size: 20px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BOTTOM NAVIGATION BAR
# ==========================================
# We place this at the top for Streamlit, but styled to look like a tab bar
cols = st.columns(3)
with cols[0]:
    if st.button("🔄\nConvert", key="nav_convert", use_container_width=True): 
        st.session_state.nav = 'Convert'
with cols[1]:
    if st.button("🏆\nLeaderboard", key="nav_leader", use_container_width=True): 
        st.session_state.nav = 'Leaderboard'
with cols[2]:
    if st.button("👛\nMy Savings", key="nav_savings", use_container_width=True): 
        st.session_state.nav = 'Savings'

st.divider()

# ==========================================
# VIEW: LEADERBOARD
# ==========================================
if st.session_state.nav == 'Leaderboard':
    st.markdown("<h3 style='text-align: center; color: #0066cc;'>WEEKLY SAVINGS LEADERBOARD</h3>", unsafe_allow_html=True)
    
    # Toggle (Visual)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.select_slider("View Mode", options=["Friends", "Global"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Data matching screenshot
    leaders = [
        {"rank": "🥇", "name": "Priya S.", "eff": "+2.4%", "img": "👩🏽"},
        {"rank": "🥈", "name": "Rahul K.", "eff": "+1.8%", "img": "👨🏽"},
        {"rank": "🥉", "name": "Amit B.", "eff": "+1.1%", "img": "👨🏻"},
        {"rank": "2", "name": "Rant K.", "eff": "+0.8%", "img": "👩🏻"},
        {"rank": "4", "name": "Antre G.", "eff": "+0.7%", "img": "👨🏾"},
        {"rank": "5", "name": "Divya K.", "eff": "+0.5%", "img": "👩🏼"},
    ]
    
    for leader in leaders:
        with st.container(border=True):
            col_rank, col_name, col_score = st.columns([1, 3, 2])
            with col_rank:
                st.write(f"### {leader['rank']}")
            with col_name:
                st.write(f"**{leader['name']}**")
                st.caption(leader['img'])
            with col_score:
                st.markdown(f"<h4 style='color: green; text-align: right;'>{leader['eff']}</h4>", unsafe_allow_html=True)
                st.caption("Efficiency")

# ==========================================
# VIEW: MY SAVINGS
# ==========================================
elif st.session_state.nav == 'Savings':
    st.markdown("<h4 style='text-align: center;'>MY SAVINGS WALLET</h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: right;'>ℹ️</div>", unsafe_allow_html=True)
    
    # Stack of Coffees Visual
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 80px; line-height: 0.8;">☕☕☕</div>
            <div style="font-size: 80px; line-height: 0.8;">☕☕☕</div>
            <div style="font-size: 120px; font-weight: bold; color: white; text-shadow: 2px 2px 4px #000000;">
                <span style="position: relative; bottom: 80px;">19</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='text-align: center;'>You saved 19 COFFEES from your last transaction!</h3>", unsafe_allow_html=True)
    st.caption("Compared to yesterday's rate.")
    
    st.button("Share my Savings", type="primary", use_container_width=True)

# ==========================================
# VIEW: CONVERT (Main Logic)
# ==========================================
elif st.session_state.nav == 'Convert':
    
    # Fetch Data for Rates
    df = fetch_data(period="1y")
    current_rate = df['Rate'].iloc[-1]
    
    # --- UI Step 1: Input ---
    if st.session_state.trans_step == 'input':
        
        # Input Card
        with st.container(border=True):
            col_in1, col_in2 = st.columns([3, 1])
            with col_in1:
                amt = st.number_input("Amount", value=1000, label_visibility="collapsed")
            with col_in2:
                st.selectbox("Cur", ["INR"], key="curr_from", label_visibility="collapsed")
        
        st.markdown("<div style='text-align: center; color: #0066cc; font-size: 24px; margin: -10px 0;'>⬇️</div>", unsafe_allow_html=True)
        
        # Output Card
        with st.container(border=True):
            col_out1, col_out2 = st.columns([3, 1])
            with col_out1:
                st.text_input("Converted", value=f"{amt/current_rate:.2f}", disabled=True, label_visibility="collapsed")
            with col_out2:
                st.selectbox("Cur", ["EUR"], key="curr_to", label_visibility="collapsed")
            
        # Quick Picks
        st.markdown("### ")
        qp_cols = st.columns(4)
        qp_cols[0].button("💲 USD")
        qp_cols[1].button("🇦🇺 AUD")
        qp_cols[2].button("🇨🇦 CAD")
        qp_cols[3].button("🇬🇧 GBP")
        
        st.markdown("### ")
        if st.button("LOG TRANSACTION", type="primary", use_container_width=True):
            st.session_state.transaction_amt = amt
            st.session_state.trans_step = 'category'
            st.rerun()

    # --- UI Step 2: Categorize ---
    elif st.session_state.trans_step == 'category':
        st.markdown("<h3 style='text-align: center;'>CATEGORIZE YOUR TRANSACTION</h3>", unsafe_allow_html=True)
        
        # Grid of circular buttons
        c1, c2, c3 = st.columns(3)
        if c1.button("🏠\nRent"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if c2.button("🎓\nTuition"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if c3.button("✈️\nTravel"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        c4, c5, c6 = st.columns(3)
        if c4.button("🛒\nShopping"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if c5.button("👪\nRemittance"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
        if c6.button("🔄\nPuniaction"): 
            st.session_state.trans_step = 'input'
            st.session_state.nav = 'Savings'
            st.rerun()
            
        st.markdown("### ")
        st.button("CONFIRM & SEE SAVINGS", type="primary", use_container_width=True)

    # --- BELOW: Analytics (FIXED CODE HERE) ---
    st.divider()
    with st.expander("📊 Advanced Market Analysis & Prediction", expanded=True):
        
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        
        # Run Models
        with st.spinner("Running AI Prediction Models..."):
            ols_model, ols_forecast = run_ols_model(df['Rate'], forecast_days)
            arima_model, arima_forecast = run_arima_model(df['Rate'], forecast_days)
            # run_garch_model now returns the fitted object AND the variance array
            garch_fit_obj, garch_volatility = run_garch_model(df['Rate'], forecast_days)

        # Plot
        fig = create_interactive_plot(df, forecast_series=arima_forecast) 
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.subheader("Trend")
            ols_dir = "UP ↗" if ols_model.params['Lag_1'] > 1 else "DOWN ↘"
            st.info(f"Trend: **{ols_dir}**")
            
        with col_res2:
            st.subheader("Prediction")
            final_pred = arima_forecast.iloc[-1]
            st.success(f"Target: **₹{final_pred:.2f}**")
            
        # Advice - FIXED SECTION
        # We assume returns are stable enough to use the last forecasted variance
        # garch_volatility contains the variance. Sqrt gives std dev (volatility).
        risk_val = np.sqrt(garch_volatility).mean()
        
        risk_level = "Low Risk" if risk_val < 0.5 else "High Risk"
        advice = generate_trading_advice(ols_dir, risk_level, current_rate, final_pred)
        
        st.warning(f"AI Recommendation: **{advice}**")
