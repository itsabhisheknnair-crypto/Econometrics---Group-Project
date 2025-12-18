import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
from datetime import timedelta
import warnings
from itertools import product

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def fetch_data():
    """Fetch and prepare exchange rate data"""
    print("=" * 60)
    print("DOWNLOADING DATA (2000-2025)")
    print("=" * 60)
    
    ticker = "EURINR=X"  # EUR to INR rate
    data = yf.download(ticker, start="2000-01-01", end="2025-12-31", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data[['Close']].copy()
    df = df.ffill()  # Use ffill() instead of fillna(method='ffill')
    
    # Keep as INR per EUR (standard quotation) - no inversion needed
    df['Rate'] = df['Close']
    df = df.resample('B').last().ffill()
    
    print(f"✓ Data loaded: {len(df)} observations")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Current rate: {df['Rate'].iloc[-1]:.4f} INR per EUR\n")
    
    return df

# ==========================================
# 2. STATIONARITY TESTING
# ==========================================
def test_stationarity(series, name="Series"):
    """
    Comprehensive stationarity testing using ADF and KPSS tests.
    
    ADF: H0 = Unit root (non-stationary)
    KPSS: H0 = Stationary
    """
    print(f"\n{'=' * 60}")
    print(f"STATIONARITY TESTS: {name}")
    print("=" * 60)
    
    # Augmented Dickey-Fuller Test
    adf_result = adfuller(series, autolag='AIC')
    print(f"\n1. Augmented Dickey-Fuller Test:")
    print(f"   Test Statistic: {adf_result[0]:.4f}")
    print(f"   P-value: {adf_result[1]:.4f}")
    print(f"   Critical Values: {adf_result[4]}")
    
    if adf_result[1] < 0.05:
        print(f"   → Result: STATIONARY (reject unit root at 5% level)")
        adf_stationary = True
    else:
        print(f"   → Result: NON-STATIONARY (cannot reject unit root)")
        adf_stationary = False
    
    # KPSS Test
    kpss_result = kpss(series, regression='c', nlags='auto')
    print(f"\n2. KPSS Test:")
    print(f"   Test Statistic: {kpss_result[0]:.4f}")
    print(f"   P-value: {kpss_result[1]:.4f}")
    print(f"   Critical Values: {kpss_result[3]}")
    
    if kpss_result[1] > 0.05:
        print(f"   → Result: STATIONARY (cannot reject stationarity)")
        kpss_stationary = True
    else:
        print(f"   → Result: NON-STATIONARY (reject stationarity)")
        kpss_stationary = False
    
    # Combined interpretation
    print(f"\n3. Combined Interpretation:")
    if adf_stationary and kpss_stationary:
        print(f"   ✓ Series is STATIONARY (both tests agree)")
        is_stationary = True
    elif not adf_stationary and not kpss_stationary:
        print(f"   ✗ Series is NON-STATIONARY (both tests agree)")
        print(f"   → Recommendation: Use differencing or cointegration")
        is_stationary = False
    else:
        print(f"   ⚠ Tests disagree - inconclusive")
        print(f"   → Recommendation: Proceed with caution, consider differencing")
        is_stationary = False
    
    return is_stationary, adf_result[1], kpss_result[1]

# ==========================================
# 3. MODEL SELECTION & DIAGNOSTICS
# ==========================================
def select_arima_order(series, max_p=5, max_d=2, max_q=5):
    """
    Select optimal ARIMA order using AIC/BIC information criteria
    """
    print(f"\n{'=' * 60}")
    print("ARIMA ORDER SELECTION (Grid Search)")
    print("=" * 60)
    
    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = None
    best_order_bic = None
    
    # Grid search
    orders = list(product(range(max_p+1), range(max_d+1), range(max_q+1)))
    
    results = []
    count = 0
    for order in orders:
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            
            results.append({
                'order': order,
                'aic': model_fit.aic,
                'bic': model_fit.bic
            })
            
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order_aic = order
            
            if model_fit.bic < best_bic:
                best_bic = model_fit.bic
                best_order_bic = order
            
            count += 1
            if count % 20 == 0:
                print(f"  Tested {count} models...", end='\r')
                
        except:
            continue
    
    print(f"\nBest order by AIC: {best_order_aic} (AIC = {best_aic:.2f})")
    print(f"Best order by BIC: {best_order_bic} (BIC = {best_bic:.2f})")
    
    # Use BIC as it penalizes complexity more
    return best_order_bic, results

def diagnostic_tests(residuals, lags=10):
    """
    Perform residual diagnostic tests
    """
    print(f"\n{'=' * 60}")
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Ljung-Box Test for Autocorrelation
    print(f"\n1. Ljung-Box Test (Autocorrelation):")
    lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)
    print(f"   P-values at lags 1, 5, 10:")
    for lag in [1, 5, 10]:
        if lag <= len(lb_test):
            pval = lb_test.iloc[lag-1, 1]  # Use iloc instead of loc
            print(f"   Lag {lag}: p-value = {pval:.4f}", end="")
            if pval > 0.05:
                print(" ✓ (No autocorrelation)")
            else:
                print(" ✗ (Autocorrelation detected)")
    
    # 2. ARCH Test for Heteroskedasticity
    print(f"\n2. ARCH Test (Heteroskedasticity):")
    arch_test = het_arch(residuals, nlags=10)
    print(f"   Test Statistic: {arch_test[0]:.4f}")
    print(f"   P-value: {arch_test[1]:.4f}")
    if arch_test[1] > 0.05:
        print(f"   ✓ No ARCH effects (homoskedastic)")
    else:
        print(f"   ✗ ARCH effects present (heteroskedastic)")
        print(f"   → Consider GARCH modeling")
    
    # 3. Normality
    print(f"\n3. Jarque-Bera Test (Normality):")
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"   Test Statistic: {jb_stat:.4f}")
    print(f"   P-value: {jb_pval:.4f}")
    if jb_pval > 0.05:
        print(f"   ✓ Residuals are normally distributed")
    else:
        print(f"   ⚠ Residuals are not normally distributed")
    
    return lb_test, arch_test

# ==========================================
# 4. MODELS WITH PROPER VALIDATION
# ==========================================
def run_ols_model(train_series, test_series, forecast_steps):
    """
    OLS with proper AR structure (only if series is stationary)
    """
    print(f"\n{'=' * 60}")
    print("OLS MODEL (AR)")
    print("=" * 60)
    
    # Create lagged features
    df_ols = pd.DataFrame({'Rate': train_series.values}, index=train_series.index)
    df_ols['Lag_1'] = df_ols['Rate'].shift(1)
    df_ols.dropna(inplace=True)
    
    X_train = sm.add_constant(df_ols['Lag_1'])
    y_train = df_ols['Rate']
    
    model = sm.OLS(y_train, X_train).fit()
    
    print("\nModel Summary:")
    print(model.summary().tables[1])
    
    # Out-of-sample validation
    test_pred = []
    for i in range(len(test_series)):
        if i == 0:
            lag_val = train_series.iloc[-1]
        else:
            lag_val = test_series.iloc[i-1]
        
        pred = model.params['const'] + model.params['Lag_1'] * lag_val
        test_pred.append(pred)
    
    # Future forecast
    last_val = test_series.iloc[-1] if len(test_series) > 0 else train_series.iloc[-1]
    forecast = []
    for _ in range(forecast_steps):
        pred = model.params['const'] + model.params['Lag_1'] * last_val
        forecast.append(pred)
        last_val = pred
    
    return model, forecast, test_pred

def run_arima_model(train_series, test_series, order, forecast_steps):
    """
    ARIMA with optimal order
    """
    print(f"\n{'=' * 60}")
    print(f"ARIMA MODEL {order}")
    print("=" * 60)
    
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    
    print("\nModel Summary:")
    print(model_fit.summary().tables[1])
    
    # Diagnostic tests
    residuals = model_fit.resid
    diagnostic_tests(residuals)
    
    # Out-of-sample validation (simplified)
    test_pred = model_fit.forecast(steps=len(test_series))
    
    # Future forecast with confidence intervals
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()
    
    return model_fit, forecast_mean, forecast_ci, test_pred.values

def run_garch_model(train_series, test_series, forecast_steps):
    """
    GARCH(1,1) for volatility forecasting
    """
    print(f"\n{'=' * 60}")
    print("GARCH(1,1) MODEL (Volatility)")
    print("=" * 60)
    
    # Calculate returns
    returns = 100 * train_series.pct_change().dropna()
    
    # Fit GARCH
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
    model_fit = model.fit(disp='off')
    
    print("\nModel Summary:")
    print(model_fit.summary().tables[1])
    
    # Forecast volatility
    forecast_res = model_fit.forecast(horizon=forecast_steps, reindex=False)
    variance_forecast = forecast_res.variance.values[-1]
    volatility_forecast = np.sqrt(variance_forecast)
    
    return model_fit, volatility_forecast

# ==========================================
# 5. EVALUATION METRICS
# ==========================================
def evaluate_forecasts(actual, predicted, model_name):
    """
    Calculate forecast accuracy metrics
    """
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Settings
    FORECAST_DAYS = 90  # 3 months
    TRAIN_RATIO = 0.8
    
    # 1. Load Data
    df = fetch_data()
    
    # 2. Test Stationarity
    is_stationary, adf_p, kpss_p = test_stationarity(df['Rate'], "Level Series")
    
    # If non-stationary, test first difference
    if not is_stationary:
        df['Rate_diff'] = df['Rate'].diff()
        is_diff_stationary, _, _ = test_stationarity(df['Rate_diff'].dropna(), "First Difference")
    
    # 3. Split Data
    split_point = int(len(df) * TRAIN_RATIO)
    train = df['Rate'].iloc[:split_point]
    test = df['Rate'].iloc[split_point:]
    
    print(f"\n{'=' * 60}")
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Training set: {len(train)} observations ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"Test set: {len(test)} observations ({test.index[0].date()} to {test.index[-1].date()})")
    
    # 4. Select ARIMA Order
    best_order, order_results = select_arima_order(train, max_p=5, max_d=2, max_q=3)
    
    # 5. Run Models
    # ARIMA (primary model)
    arima_model, arima_forecast, arima_ci, arima_test_pred = run_arima_model(
        train, test, best_order, FORECAST_DAYS
    )
    
    # OLS (for comparison)
    ols_model, ols_forecast, ols_test_pred = run_ols_model(train, test, FORECAST_DAYS)
    
    # GARCH (volatility)
    garch_model, garch_volatility = run_garch_model(train, test, FORECAST_DAYS)
    
    # 6. Evaluate on Test Set
    print(f"\n{'=' * 60}")
    print("OUT-OF-SAMPLE FORECAST EVALUATION")
    print("=" * 60)
    
    arima_metrics = evaluate_forecasts(test.values, arima_test_pred, "ARIMA")
    ols_metrics = evaluate_forecasts(test.values, ols_test_pred, "OLS")
    
    # 7. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Historical + Forecasts
    ax1 = axes[0]
    history_subset = df['Rate'].iloc[-500:]
    ax1.plot(history_subset.index, history_subset, label='Historical', color='black', linewidth=1.5)
    ax1.axvline(test.index[0], color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
    
    # Forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=FORECAST_DAYS)
    
    # ARIMA forecast with confidence intervals
    ax1.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', 
             linestyle='-', color='red', linewidth=2)
    ax1.fill_between(forecast_dates, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], 
                      alpha=0.2, color='red', label='95% Confidence Interval')
    
    # OLS forecast
    ax1.plot(forecast_dates, ols_forecast, label='OLS Forecast', 
             linestyle='--', color='blue', linewidth=1.5, alpha=0.7)
    
    ax1.set_title(f"INR/EUR Exchange Rate: {FORECAST_DAYS}-Day Forecast", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Exchange Rate (INR per EUR)")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Out-of-Sample Performance
    ax2 = axes[1]
    ax2.plot(test.index, test.values, label='Actual', color='black', linewidth=2)
    ax2.plot(test.index, arima_test_pred, label='ARIMA Predictions', 
             linestyle='-', color='red', alpha=0.7)
    ax2.plot(test.index, ols_test_pred, label='OLS Predictions', 
             linestyle='--', color='blue', alpha=0.7)
    
    ax2.set_title("Out-of-Sample Forecast Performance", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Exchange Rate (INR per EUR)")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("econometric_forecast_analysis.png", dpi=300, bbox_inches='tight')
    
    # 8. Final Summary
    print(f"\n{'=' * 60}")
    print("FORECAST SUMMARY")
    print("=" * 60)
    print(f"\nCurrent Rate (Latest): {df['Rate'].iloc[-1]:.4f} INR per EUR")
    print(f"\nForecasts for {FORECAST_DAYS} days ahead:")
    print(f"  ARIMA {best_order}: {arima_forecast.iloc[-1]:.4f} INR per EUR")
    print(f"  95% CI: [{arima_ci.iloc[-1, 0]:.4f}, {arima_ci.iloc[-1, 1]:.4f}]")
    print(f"  OLS: {ols_forecast[-1]:.4f} INR per EUR")
    
    print(f"\nExpected Daily Volatility: {garch_volatility.mean():.4f}%")
    
    print(f"\n{'=' * 60}")
    print("✓ Analysis complete. Plot saved as 'econometric_forecast_analysis.png'")
    print("=" * 60)
