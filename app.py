import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Forex Fee Forecaster",
    page_icon="💱",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & MODELS
# -----------------------------------------------------------------------------

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    cumulative = prices / prices.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100

def calculate_metrics(prices):
    """Calculate various performance metrics."""
    returns = prices.pct_change().dropna() * 100
    metrics = {
        'Mean Return (%)': returns.mean(),
        'Volatility (%)': returns.std(),
        'Sharpe Ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'Maximum Drawdown (%)': calculate_max_drawdown(prices),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Mean Price': prices.mean(),
        'Price Std Dev': prices.std(),
        'Min Price': prices.min(),
        'Max Price': prices.max()
    }
    return metrics

def run_ols_model(prices, forecast_days=30):
    """OLS model for trend forecasting."""
    try:
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        X_const = add_constant(X)
        model = OLS(y, X_const)
        results = model.fit()
        
        X_future = np.arange(len(prices), len(prices) + forecast_days).reshape(-1, 1)
        X_future_const = add_constant(X_future)
        forecast = results.predict(X_future_const)
        
        # Calculate standard error for confidence intervals
        mse = results.mse_resid
        std_error = np.sqrt(mse * (1 + 1/len(X) + (X_future - X.mean())**2 / ((X - X.mean())**2).sum()))
        
        return {
            'forecast': forecast,
            'r_squared': results.rsquared,
            'trend_coef': results.params[1],
            'std_error': std_error.flatten(),
            'trend': 'Upward' if results.params[1] > 0 else 'Downward',
            'model': results
        }
    except Exception as e:
        st.error(f"OLS Model Error: {e}")
        return None

def run_arima_model(prices, forecast_days=30, order=(1,1,1)):
    """ARIMA model for time series forecasting."""
    try:
        # Simple grid search (simplified for speed in Streamlit)
        best_aic = np.inf
        best_order = order
        best_model = None
        
        # Reduced grid for performance
        for p in range(0, 2):
            for d in range(0, 2):
                for q in range(0, 2):
                    try:
                        model = ARIMA(prices, order=(p,d,q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p,d,q)
                            best_model = results
                    except:
                        continue
        
        if best_model is None:
            model = ARIMA(prices, order=order)
            best_model = model.fit()
            best_order = order
        
        forecast = best_model.forecast(steps=forecast_days)
        
        return {
            'forecast': forecast.values,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'order': best_order,
            'model': best_model
        }
    except Exception as e:
        # Fallback
        last_value = prices.iloc[-1]
        forecast = np.array([last_value] * forecast_days)
        return {
            'forecast': forecast,
            'aic': np.nan,
            'bic': np.nan,
            'order': order,
            'model': None
        }

def run_garch_model(returns, forecast_days=30):
    """GARCH model for volatility forecasting."""
    try:
        # Rescale returns if volatility is very small to avoid convergence issues
        scale_factor = 100 if returns.std() < 0.01 else 1
        scaled_returns = returns * scale_factor
        
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        
        forecast = results.forecast(horizon=forecast_days)
        conditional_vol = forecast.variance.values[-1, :]
        
        # Rescale volatility back
        final_vol = np.sqrt(conditional_vol) / scale_factor
        
        return {
            'volatility': final_vol,
            'persistence': results.params.get('alpha[1]', 0) + results.params.get('beta[1]', 0)
        }
    except Exception as e:
        st.warning(f"GARCH model failed to converge: {e}")
        volatility = np.array([returns.std()] * forecast_days)
        return {
            'volatility': volatility,
            'persistence': np.nan
        }

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input("Ticker Symbol", value="EURUSD=X", help="Yahoo Finance Ticker (e.g., GBPUSD=X, JPY=X)")
        
        period = st.selectbox("Historical Data Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        
        forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30)
        
        st.divider()
        st.subheader("🎓 Payment Simulator")
        tuition_amount = st.number_input("Payment Amount", value=20000, step=500, help="Amount in target currency or base currency depending on perspective")
        
        if st.button("Run Forecast Analysis", type="primary"):
            st.session_state['run_analysis'] = True

    # --- Title Section ---
    st.title("💱 Forex Fee Forecaster")
    st.markdown(f"Forecasting trends and volatility for **{ticker}** to optimize international payments.")

    # --- Data Loading ---
    if st.session_state.get('run_analysis', False):
        with st.spinner('Downloading market data...'):
            try:
                # Calculate start date based on period
                end_date = datetime.now()
                if period == "3mo": start_date = end_date - timedelta(days=90)
                elif period == "6mo": start_date = end_date - timedelta(days=180)
                elif period == "1y": start_date = end_date - timedelta(days=365)
                elif period == "2y": start_date = end_date - timedelta(days=730)
                else: start_date = end_date - timedelta(days=365*5)
                
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    st.error(f"No data found for ticker {ticker}. Please check the symbol.")
                    st.stop()
                    
                prices = df['Close'].dropna()
                # Handle MultiIndex if present (yfinance update)
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]
                
                returns = prices.pct_change().dropna() * 100
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        # --- Dashboard Layout ---
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Overview", "🤖 Forecast Models", "⚡ Volatility (GARCH)", "💡 Recommendation"])

        # TAB 1: MARKET OVERVIEW
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            current_price = prices.iloc[-1]
            prev_price = prices.iloc[-2]
            delta = current_price - prev_price
            
            col1.metric("Current Rate", f"{current_price:.4f}", f"{delta:.4f}")
            col2.metric("Period High", f"{prices.max():.4f}")
            col3.metric("Period Low", f"{prices.min():.4f}")
            col4.metric("Volatility (Std)", f"{returns.std():.2f}%")
            
            # Interactive Price Chart
            fig_price = px.line(x=prices.index, y=prices.values, title=f"{ticker} Historical Prices")
            fig_price.update_layout(xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Metrics DataFrame
            st.subheader("Descriptive Statistics")
            metrics = calculate_metrics(prices)
            st.json(metrics)

        # TAB 2: FORECAST MODELS (OLS & ARIMA)
        with tab2:
            st.subheader("Trend Analysis")
            
            # Run Models
            with st.spinner("Running OLS and ARIMA models..."):
                ols_res = run_ols_model(prices, forecast_days)
                arima_res = run_arima_model(prices, forecast_days)
            
            col_a, col_b = st.columns(2)
            
            # OLS Results
            with col_a:
                st.markdown("### OLS Linear Trend")
                if ols_res:
                    st.info(f"Trend: **{ols_res['trend']}** (R²: {ols_res['r_squared']:.4f})")
                    
                    # Plot OLS
                    future_dates = [prices.index[-1] + timedelta(days=x) for x in range(1, forecast_days + 1)]
                    fig_ols = go.Figure()
                    fig_ols.add_trace(go.Scatter(x=prices.index, y=prices.values, name='Historical'))
                    fig_ols.add_trace(go.Scatter(x=future_dates, y=ols_res['forecast'], name='Forecast', line=dict(color='orange', dash='dash')))
                    fig_ols.update_layout(title="OLS Linear Forecast")
                    st.plotly_chart(fig_ols, use_container_width=True)

            # ARIMA Results
            with col_b:
                st.markdown("### ARIMA Time Series")
                if arima_res:
                    st.info(f"Best Order: {arima_res['order']} (AIC: {arima_res['aic']:.2f})")
                    
                    # Plot ARIMA
                    future_dates = [prices.index[-1] + timedelta(days=x) for x in range(1, forecast_days + 1)]
                    fig_arima = go.Figure()
                    fig_arima.add_trace(go.Scatter(x=prices.index, y=prices.values, name='Historical'))
                    fig_arima.add_trace(go.Scatter(x=future_dates, y=arima_res['forecast'], name='Forecast', line=dict(color='green', dash='dot')))
                    fig_arima.update_layout(title="ARIMA Forecast")
                    st.plotly_chart(fig_arima, use_container_width=True)

        # TAB 3: VOLATILITY (GARCH)
        with tab3:
            st.subheader("Volatility Forecasting (GARCH)")
            
            with st.spinner("Running GARCH model..."):
                garch_res = run_garch_model(returns, forecast_days)
            
            if garch_res:
                col_v1, col_v2 = st.columns([1, 2])
                with col_v1:
                    st.metric("Forecasted Volatility (30d avg)", f"{garch_res['volatility'].mean():.4f}%")
                    st.metric("Persistence", f"{garch_res.get('persistence', 0):.4f}")
                
                with col_v2:
                    future_dates = [prices.index[-1] + timedelta(days=x) for x in range(1, forecast_days + 1)]
                    fig_vol = px.line(x=future_dates, y=garch_res['volatility'], title="Forecasted Volatility Horizon")
                    fig_vol.update_yaxes(title="Volatility (%)")
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                st.markdown("""
                **Interpretation:**
                * **High Volatility:** Expect large price swings. Consider hedging or locking in rates.
                * **Low Volatility:** Market is stable. Standard spot payments may be safer.
                """)

        # TAB 4: RECOMMENDATION
        with tab4:
            st.header("Decision Support")
            
            avg_forecast_price = np.mean([ols_res['forecast'][-1], arima_res['forecast'][-1]])
            current_rate = prices.iloc[-1]
            
            # Simple tuition logic: Assuming Tuition is in Base Currency (e.g., USD) and we hold Quote Currency
            # OR Tuition is in Quote Currency and we hold Base Currency.
            # For simplicity: "Cost to buy 'tuition_amount' of Quote Currency using Base Currency"
            
            # Scenario: You need to pay 'tuition_amount' (e.g., 20k EUR). Ticker EURUSD=X.
            # Rate is USD per EUR.
            # Cost in USD = Amount * Rate
            
            current_cost = tuition_amount * current_rate
            forecast_cost = tuition_amount * avg_forecast_price
            diff = forecast_cost - current_cost
            pct_change = (diff / current_cost) * 100
            
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Current Cost", f"{current_cost:,.2f}")
            col_d2.metric("Forecasted Cost (30 Days)", f"{forecast_cost:,.2f}")
            col_d3.metric("Projected Difference", f"{diff:,.2f}", f"{pct_change:.2f}%")
            
            st.divider()
            
            # Recommendation Logic
            if pct_change > 0.5:
                st.success("🟢 **Recommendation: PAY NOW**")
                st.write(f"Models predict the rate will increase by {pct_change:.2f}%. Paying now could save you approximately {diff:,.2f}.")
            elif pct_change < -0.5:
                st.warning("🔵 **Recommendation: WAIT**")
                st.write(f"Models predict the rate will decrease by {abs(pct_change):.2f}%. Waiting could save you approximately {abs(diff):,.2f}.")
            else:
                st.info("🟡 **Recommendation: HOLD / NEUTRAL**")
                st.write("Forecast indicates relatively stable prices. Monitor volatility or use dollar-cost averaging.")

    else:
        st.info("👈 Select parameters in the sidebar and click **Run Forecast Analysis** to begin.")

if __name__ == "__main__":
    main()
