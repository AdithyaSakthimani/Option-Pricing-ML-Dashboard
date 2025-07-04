# Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from arch import arch_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Additional imports for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ARIMA_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. ARIMA functionality will be disabled.")
    ARIMA_AVAILABLE = False

# NEW: ARIMA-based spot price prediction
def predict_spot_price_arima(ticker, evaluation_date, forecast_days=1, lookback_days=252, 
                           use_garch=False, confidence_level=0.95):
    """
    Predict spot price using ARIMA (optionally with GARCH) for given evaluation date.
    
    Parameters:
    - ticker: Stock ticker symbol
    - evaluation_date: Date for which to predict the spot price
    - forecast_days: Number of days ahead to forecast (default 1)
    - lookback_days: Number of historical days to use for training
    - use_garch: Whether to use ARIMA+GARCH (True) or just ARIMA (False)
    - confidence_level: Confidence level for prediction intervals
    
    Returns:
    - Dictionary with predicted price, confidence intervals, and model info
    """
    
    if not ARIMA_AVAILABLE:
        return {'error': 'ARIMA functionality not available. Install statsmodels.'}
    
    try:
        evaluation_date = pd.to_datetime(evaluation_date)
        # Get data up to evaluation date (no future leakage)
        end_date = evaluation_date - pd.Timedelta(days=1)  # Day before evaluation
        start_date = end_date - pd.Timedelta(days=lookback_days + 30)  # Extra buffer
        
        print(f"Fetching data from {start_date.date()} to {end_date.date()} for {ticker}")
        
        # Fetch historical data
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), 
                          progress=False)
        
        if data.empty or len(data) < 30:
            return {'error': f'Insufficient historical data for {ticker}'}
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        # Get closing prices and ensure we have enough data
        prices = data['Close'].dropna()
        if len(prices) < 30:
            return {'error': f'Insufficient price data for {ticker}'}
        
        # Use only the most recent lookback_days
        prices = prices.tail(lookback_days)
        
        # Check for stationarity and difference if needed
        def check_stationarity(ts, significance=0.05):
            result = adfuller(ts.dropna())
            return result[1] <= significance  # p-value <= significance level
        
        # Determine differencing order
        original_prices = prices.copy()
        diff_order = 0
        current_series = prices
        
        # Test up to 2nd order differencing
        for d in range(3):
            if check_stationarity(current_series):
                diff_order = d
                break
            if d < 2:  # Don't difference beyond 2nd order
                current_series = current_series.diff().dropna()
        
        print(f"Using differencing order: {diff_order}")
        
        # Auto-select ARIMA parameters using AIC
        def select_arima_order(ts, max_p=3, max_q=3):
            best_aic = np.inf
            best_order = (1, diff_order, 1)
            
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, diff_order, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, diff_order, q)
                    except:
                        continue
            
            return best_order
        
        # Select best ARIMA order
        best_order = select_arima_order(prices)
        print(f"Selected ARIMA order: {best_order}")
        
        if not use_garch:
            # Simple ARIMA model
            model = ARIMA(prices, order=best_order)
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int(alpha=1-confidence_level)
            
            predicted_price = float(forecast.iloc[-1])  # Last forecasted value
            lower_ci = float(forecast_ci.iloc[-1, 0])
            upper_ci = float(forecast_ci.iloc[-1, 1])
            
            model_info = {
                'model_type': 'ARIMA',
                'order': best_order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        else:
            # ARIMA + GARCH model
            # First fit ARIMA for the mean
            arima_model = ARIMA(prices, order=best_order)
            arima_fitted = arima_model.fit()
            
            # Get residuals for GARCH
            residuals = arima_fitted.resid
            
            # Fit GARCH model to residuals
            garch_model = arch_model(residuals * 100, vol='Garch', p=1, q=1)  # Scale up for GARCH
            garch_fitted = garch_model.fit(disp='off')
            
            # Forecast mean (ARIMA) and variance (GARCH)
            arima_forecast = arima_fitted.forecast(steps=forecast_days)
            garch_forecast = garch_fitted.forecast(horizon=forecast_days)
            
            predicted_price = float(arima_forecast.iloc[-1])
            predicted_variance = float(garch_forecast.variance.iloc[-1, -1]) / 10000  # Scale back
            predicted_std = np.sqrt(predicted_variance)
            
            # Calculate confidence intervals using predicted volatility
            z_score = norm.ppf((1 + confidence_level) / 2)
            lower_ci = predicted_price - z_score * predicted_std
            upper_ci = predicted_price + z_score * predicted_std
            
            model_info = {
                'model_type': 'ARIMA+GARCH',
                'arima_order': best_order,
                'arima_aic': arima_fitted.aic,
                'garch_aic': garch_fitted.aic
            }
        
        # Ensure predicted price is positive
        predicted_price = max(predicted_price, 0.01)
        lower_ci = max(lower_ci, 0.01)
        upper_ci = max(upper_ci, 0.01)
        
        # Get actual last known price for comparison
        last_known_price = float(prices.iloc[-1])
        
        results = {
            'predicted_price': round(predicted_price, 2),
            'confidence_interval': {
                'lower': round(lower_ci, 2),
                'upper': round(upper_ci, 2),
                'confidence_level': confidence_level
            },
            'last_known_price': round(last_known_price, 2),
            'price_change': round(predicted_price - last_known_price, 2),
            'price_change_pct': round((predicted_price - last_known_price) / last_known_price * 100, 2),
            'model_info': model_info,
            'data_points_used': len(prices),
            'forecast_date': evaluation_date.strftime('%Y-%m-%d')
        }
        
        print(f"ARIMA prediction successful: ${predicted_price:.2f} "
              f"[{lower_ci:.2f}, {upper_ci:.2f}] for {ticker}")
        
        return results
        
    except Exception as e:
        print(f"ARIMA prediction failed for {ticker}: {str(e)}")
        return {'error': f'ARIMA prediction failed: {str(e)}'}

# MODIFIED: Enhanced get_spot_price function with ARIMA option
def get_spot_price(ticker, date, predict_spot=False, use_garch=False):
    """
    Get spot price for a specific date with optional ARIMA prediction.
    
    Parameters:
    - ticker: Stock symbol
    - date: Target date
    - predict_spot: If True, use ARIMA prediction; if False, use Yahoo Finance
    - use_garch: If True and predict_spot=True, use ARIMA+GARCH
    
    Returns:
    - Float: spot price, or dict with prediction details if predict_spot=True
    """
    
    if predict_spot:
        # Use ARIMA prediction
        arima_result = predict_spot_price_arima(ticker, date, use_garch=use_garch)
        
        if 'error' not in arima_result:
            if isinstance(arima_result, dict) and 'predicted_price' in arima_result:
                return arima_result  # Return full prediction details
            else:
                return arima_result.get('predicted_price', np.nan)
        else:
            print(f"ARIMA failed, falling back to Yahoo Finance for {ticker}")
            # Fallback to Yahoo Finance
            return get_spot_price_yahoo(ticker, date)
    else:
        # Use original Yahoo Finance method
        return get_spot_price_yahoo(ticker, date)

def get_spot_price_yahoo(ticker, date):
    """Original Yahoo Finance spot price function"""
    try:
        date = pd.to_datetime(date)
        start_date = date - pd.Timedelta(days=5)
        end_date = date + pd.Timedelta(days=5)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return np.nan
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
            
        # Get closest available date
        closest_date = min(data.index, key=lambda x: abs(x.date() - date.date()))
        return float(data.loc[closest_date, 'Close'])  # Ensure scalar return
    except:
        return np.nan

# Step 1: Forecast GARCH volatility
def forecast_garch_volatility(ticker, end_date, horizon_days=1, window=60):
    """Forecast volatility using GARCH model"""
    try:
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.Timedelta(days=window * 2)
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        
        if data.empty or len(data) < 30:
            return np.nan
            
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        close_prices = data['Close']
        returns = 100 * close_prices.pct_change().dropna()
        
        if len(returns) < 20:
            return np.nan
            
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=horizon_days)
        var_forecast = forecast.variance.iloc[-1].values[0]  # Fixed: get scalar value
        vol_forecast = np.sqrt(var_forecast) / 100
        return vol_forecast
    except Exception as e:
        print(f"GARCH failed for {ticker} on {end_date}: {e}")
        return np.nan

# Step 2: Calculate Greeks
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks using Black-Scholes formulas"""
    if any(pd.isna(x) for x in [S, K, T, r, sigma]) or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {'Delta': np.nan, 'Gamma': np.nan, 'Vega': np.nan, 'Theta': np.nan, 'Rho': np.nan}

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                      r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                     r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        theta = theta_call if option_type == 'call' else theta_put
        rho = rho_call if option_type == 'call' else rho_put

        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}
    except:
        return {'Delta': np.nan, 'Gamma': np.nan, 'Vega': np.nan, 'Theta': np.nan, 'Rho': np.nan}

# Step 3: Fetch options data
def get_option_data(ticker, max_expiries=3):
    """Fetch options data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options[:max_expiries]
        option_data = []

        for exp in expiries:
            try:
                chain = stock.option_chain(exp)
                for df, typ in [(chain.calls, 'call'), (chain.puts, 'put')]:
                    df = df.copy()
                    df['option_type'] = typ
                    df['expiration_date'] = pd.to_datetime(exp)
                    df['stock'] = ticker
                    df['lastTradeDate'] = pd.to_datetime(datetime.today().date())
                    option_data.append(df)
            except Exception as e:
                print(f"Failed for {ticker} - {exp}: {e}")
                continue
        
        return pd.concat(option_data, ignore_index=True) if option_data else pd.DataFrame()
    except Exception as e:
        print(f"Failed to get options data for {ticker}: {e}")
        return pd.DataFrame()

# Step 4: Get historical volatility
def get_historical_volatility(ticker, end_date, window=30):
    """Calculate historical volatility"""
    try:
        end_date = pd.to_datetime(end_date)
        start_date = end_date - pd.Timedelta(days=window + 10)
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        
        if data.empty or len(data) < 10:
            return np.nan
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
            
        returns = data['Close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility
    except:
        return np.nan

# MODIFIED: Prepare features with spot price prediction option
def prepare_features(df, r=0.05, predict_spot=False, use_garch=False):
    """Prepare features with improved data cleaning and feature engineering"""
    print(f"Initial data: {len(df)} records")
    
    if df.empty:
        return df
    
    # Calculate time to expiry
    df['time_to_expiry'] = (df['expiration_date'] - df['lastTradeDate']).dt.days / 365
    
    # Filter out invalid data
    initial_count = len(df)
    df = df[df['time_to_expiry'] > 1/365]  # At least 1 day to expiry
    df = df[df['time_to_expiry'] < 2]      # Less than 2 years
    df = df[df['lastPrice'] > 0.01]        # Minimum price threshold
    df = df[df['lastPrice'] < 1000]        # Maximum price threshold
    df = df[df['strike'] > 0]              # Valid strike prices
    
    print(f"After basic filtering: {len(df)} records (removed {initial_count - len(df)})")
    
    if df.empty:
        return df
    
    # Get spot prices (with optional ARIMA prediction)
    print(f"Fetching spot prices (predict_spot={predict_spot})...")
    if predict_spot:
        # Store prediction details for analysis
        df['spot_price_details'] = df.apply(
            lambda row: get_spot_price(row['stock'], row['lastTradeDate'], 
                                     predict_spot=True, use_garch=use_garch), axis=1
        )
        
        # Extract the predicted price
        def extract_predicted_price(details):
            if isinstance(details, dict):
                if 'predicted_price' in details:
                    return details['predicted_price']
                elif 'error' in details:
                    return np.nan
            return details if pd.notna(details) else np.nan
        
        df['spot_price'] = df['spot_price_details'].apply(extract_predicted_price)
    else:
        df['spot_price'] = df.apply(lambda row: get_spot_price(row['stock'], row['lastTradeDate']), axis=1)
    
    df = df.dropna(subset=['spot_price'])
    df = df[df['spot_price'] > 0]
    
    print(f"After spot price filtering: {len(df)} records")
    
    if df.empty:
        return df
    
    # Handle volatility
    print("Processing volatilities...")
    def get_volatility(row):
        # Use implied volatility if available and reasonable
        if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']):
            iv = float(row['impliedVolatility'])  # Ensure scalar
            if 0.01 <= iv <= 5.0:  # Reasonable range for IV
                return iv
        
        # Fallback to GARCH for short-term options
        if row['time_to_expiry'] <= 30/365:  # 30 days or less
            garch_vol = forecast_garch_volatility(row['stock'], row['lastTradeDate'])
            if pd.notna(garch_vol) and 0.01 <= garch_vol <= 5.0:
                return garch_vol
        
        # Fallback to historical volatility
        hist_vol = get_historical_volatility(row['stock'], row['lastTradeDate'])
        if pd.notna(hist_vol) and 0.01 <= hist_vol <= 5.0:
            return hist_vol
        
        # Final fallback
        return 0.25
    
    df['volatility'] = df.apply(get_volatility, axis=1)
    
    # Calculate Greeks
    print("Calculating Greeks...")
    greeks_list = []
    for _, row in df.iterrows():
        greeks = calculate_greeks(
            S=float(row['spot_price']),
            K=float(row['strike']),
            T=float(row['time_to_expiry']),
            r=r,
            sigma=float(row['volatility']),
            option_type=row['option_type']
        )
        greeks_list.append(greeks)
    
    greeks_df = pd.DataFrame(greeks_list)
    df = pd.concat([df.reset_index(drop=True), greeks_df], axis=1)
    
    # Remove records with invalid Greeks
    df = df.dropna(subset=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
    
    # Feature engineering
    df['moneyness'] = df['spot_price'] / df['strike']
    df['log_moneyness'] = np.log(df['moneyness'])
    df['vol_time'] = df['volatility'] * np.sqrt(df['time_to_expiry'])
    df['intrinsic_value'] = np.where(
        df['option_type'] == 'call',
        np.maximum(df['spot_price'] - df['strike'], 0),
        np.maximum(df['strike'] - df['spot_price'], 0)
    )
    
    print(f"Final prepared data: {len(df)} records")
    
    return df

# Step 7: Train improved model
def train_model(df):
    """Train model with proper feature scaling and validation"""
    print("=== TRAINING MODEL ===")
    
    if len(df) < 50:
        print("ERROR: Insufficient training data. Need at least 50 records.")
        return None, None, None, None, None
    
    # Define features
    base_features = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'strike', 'spot_price', 'time_to_expiry']
    engineered_features = ['moneyness', 'log_moneyness', 'vol_time', 'volatility', 'intrinsic_value']
    features = base_features + engineered_features
    
    # Prepare data
    X = df[features].copy()
    y = df['lastPrice'].copy()
    
    # Remove any remaining NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Training on {len(X)} samples with {len(features)} features")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 FEATURE IMPORTANCE ===")
    print(feature_importance.head(10))
    
    return model, scaler, features, y_pred, y_test

# Step 8: Black-Scholes price calculation
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)  # Price cannot be negative
    except:
        return np.nan

# MODIFIED: Test model prediction with ARIMA spot price option
def test_model_prediction(model, scaler, features, stock_ticker, strike_price, option_type, 
                         expiry_date, evaluation_date, risk_free_rate=0.05, 
                         predict_spot=False, use_garch=False):
    """Test model prediction with optional ARIMA spot price prediction"""
    
    try:
        # Convert dates
        expiry_date = pd.to_datetime(expiry_date)
        evaluation_date = pd.to_datetime(evaluation_date)
        time_to_expiry = (expiry_date - evaluation_date).days / 365.0
        
        if time_to_expiry <= 0:
            return {'error': 'Option has expired or expires today'}
        
        # Get spot price (with optional ARIMA prediction)
        if predict_spot:
            print(f"Using ARIMA prediction for spot price (GARCH: {use_garch})")
            spot_result = get_spot_price(stock_ticker, evaluation_date, 
                                       predict_spot=True, use_garch=use_garch)
            
            if isinstance(spot_result, dict):
                if 'predicted_price' in spot_result:
                    spot_price = float(spot_result['predicted_price'])
                    spot_price_info = spot_result
                elif 'error' in spot_result:
                    return {'error': f'ARIMA spot price prediction failed: {spot_result["error"]}'}
                else:
                    spot_price = float(spot_result)
                    spot_price_info = {'predicted_price': spot_price, 'method': 'ARIMA_fallback'}
            else:
                spot_price = float(spot_result)
                spot_price_info = {'predicted_price': spot_price, 'method': 'ARIMA_fallback'}
        else:
            print("Using Yahoo Finance for spot price")
            spot_price = get_spot_price_yahoo(stock_ticker, evaluation_date)
            if pd.isna(spot_price):
                return {'error': f'Could not get spot price for {stock_ticker}'}
            spot_price = float(spot_price)
            spot_price_info = {'actual_price': spot_price, 'method': 'Yahoo_Finance'}
        
        # Get volatility
        if time_to_expiry <= 30/365:  # Short-term: use GARCH
            volatility = forecast_garch_volatility(stock_ticker, evaluation_date)
            vol_method = "GARCH"
        else:  # Long-term: use historical
            volatility = get_historical_volatility(stock_ticker, evaluation_date)
            vol_method = "Historical"
        
        if pd.isna(volatility) or volatility <= 0:
            volatility = 0.25  # Default fallback
            vol_method = "Default"
        
        # Ensure volatility is scalar
        volatility = float(volatility)
        
        # Calculate Greeks
        greeks = calculate_greeks(
            S=spot_price, K=float(strike_price), T=float(time_to_expiry),
            r=float(risk_free_rate), sigma=volatility, option_type=option_type.lower()
        )
        
        # Check if any Greeks are NaN
        if any(pd.isna(v) for v in greeks.values()):
            return {'error': 'Failed to calculate Greeks'}
        
        # Prepare features (same order as training)
        moneyness = spot_price / strike_price
        log_moneyness = np.log(moneyness)
        vol_time = volatility * np.sqrt(time_to_expiry)
        
        if option_type.lower() == 'call':
            intrinsic_value = max(spot_price - strike_price, 0)
        else:
            intrinsic_value = max(strike_price - spot_price, 0)
        
        # Create feature array - ensure all values are scalars
        feature_values = [
            float(greeks['Delta']), float(greeks['Gamma']), float(greeks['Vega']), 
            float(greeks['Theta']), float(greeks['Rho']),
            float(strike_price), float(spot_price), float(time_to_expiry),
            float(moneyness), float(log_moneyness), float(vol_time), 
            float(volatility), float(intrinsic_value)
        ]
        
        # Scale features
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        ml_price = float(model.predict(feature_array_scaled)[0])
        ml_price = max(ml_price, 0.01)  # Ensure positive price
        
        # Calculate Black-Scholes price
        bs_price = black_scholes_price(spot_price, strike_price, time_to_expiry,
                                     risk_free_rate, volatility, option_type.lower())
        
        results = {
            'inputs': {
                'stock_ticker': stock_ticker,
                'strike_price': strike_price,
                'option_type': option_type.lower(),
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'evaluation_date': evaluation_date.strftime('%Y-%m-%d'),
                'time_to_expiry_days': (expiry_date - evaluation_date).days,
                'time_to_expiry_years': round(time_to_expiry, 4)
            },
            'market_data': {
                'spot_price': round(spot_price, 2),
                'spot_price_method': 'ARIMA_Prediction' if predict_spot else 'Yahoo_Finance',
                'volatility': round(volatility, 4),
                'volatility_method': vol_method,
                'risk_free_rate': risk_free_rate
            },
            'spot_price_details': spot_price_info,
            'greeks': {k: round(float(v), 4) for k, v in greeks.items()},
            'predictions': {
                'ml_model_price': round(ml_price, 2),
                'black_scholes_price': round(bs_price, 2),
                'price_difference': round(ml_price - bs_price, 2),
                'percentage_difference': round((ml_price - bs_price) / bs_price * 100, 2) if bs_price > 0 else 0
            }
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# MODIFIED: Main pipeline with ARIMA options
def run_pipeline(ticker, max_expiries=3, predict_spot=False, use_garch=False):
    """Run the complete options pricing pipeline with optional ARIMA spot prediction"""
    print(f"=== RUNNING PIPELINE FOR {ticker} ===")
    print(f"Spot price prediction: {'ARIMA' + ('+GARCH' if use_garch else '') if predict_spot else 'Yahoo Finance'}")
    
    # Get options data
    print("Fetching options data...")
    df = get_option_data(ticker, max_expiries)
    
    if df.empty:
        print(f"No options data found for {ticker}")
        return None, None, None, None, None
    
    # Prepare features (with optional ARIMA spot prediction)
    df = prepare_features(df, predict_spot=predict_spot, use_garch=use_garch)
    
    if df.empty or len(df) < 50:
        print(f"Insufficient quality data for {ticker}")
        return None, None, None, None, None
    
    # Train model
    model, scaler, features, y_pred, y_test = train_model(df)
    
    if model is None:
        return None, None, None, None, None
    
    print(f"Pipeline completed successfully for {ticker}")
    return model, scaler, features, y_pred, y_test

# NEW: Utility function for ARIMA model diagnostics
def diagnose_arima_model(ticker, evaluation_date, lookback_days=252, use_garch=False):
    """
    Perform comprehensive ARIMA model diagnostics and validation.
    
    Returns detailed analysis of model fit, residuals, and forecasting performance.
    """
    
    if not ARIMA_AVAILABLE:
        return {'error': 'ARIMA functionality not available. Install statsmodels.'}
    
    try:
        evaluation_date = pd.to_datetime(evaluation_date)
        end_date = evaluation_date - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=lookback_days + 60)
        
        # Get data
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), 
                          progress=False)
        
        if data.empty or len(data) < 50:
            return {'error': f'Insufficient data for {ticker}'}
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        prices = data['Close'].dropna().tail(lookback_days)
        
        # Stationarity tests
        def check_stationarity_detailed(ts):
            result = adfuller(ts.dropna())
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] <= 0.05
            }
        
        original_stationarity = check_stationarity_detailed(prices)
        
        # Determine differencing
        diff_order = 0
        current_series = prices
        stationarity_results = [original_stationarity]
        
        for d in range(1, 3):
            current_series = current_series.diff().dropna()
            stat_result = check_stationarity_detailed(current_series)
            stationarity_results.append(stat_result)
            if stat_result['is_stationary']:
                diff_order = d
                break
        
        # Fit ARIMA model
        def select_arima_detailed(ts, max_p=3, max_q=3):
            results = []
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, diff_order, q))
                        fitted = model.fit()
                        results.append({
                            'order': (p, diff_order, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic,
                            'model': fitted
                        })
                    except:
                        continue
            
            return sorted(results, key=lambda x: x['aic'])
        
        model_results = select_arima_detailed(prices)
        if not model_results:
            return {'error': 'Failed to fit any ARIMA models'}
        
        best_model = model_results[0]['model']
        best_order = model_results[0]['order']
        
        # Residual diagnostics
        residuals = best_model.resid
        
        # Ljung-Box test for residual autocorrelation
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Out-of-sample validation (last 20% of data)
        train_size = int(len(prices) * 0.8)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Fit model on training data
        train_model = ARIMA(train_data, order=best_order).fit()
        
        # Forecast test period
        forecast_steps = len(test_data)
        forecast = train_model.forecast(steps=forecast_steps)
        
        # Calculate forecast errors
        forecast_errors = test_data.values - forecast.values
        mae = np.mean(np.abs(forecast_errors))
        rmse = np.sqrt(np.mean(forecast_errors**2))
        mape = np.mean(np.abs(forecast_errors / test_data.values)) * 100
        
        # Return comprehensive diagnostics
        diagnostics = {
            'model_info': {
                'best_order': best_order,
                'aic': model_results[0]['aic'],
                'bic': model_results[0]['bic'],
                'log_likelihood': best_model.llf
            },
            'stationarity_tests': {
                'original': original_stationarity,
                'differencing_order': diff_order,
                'final_stationary': stationarity_results[diff_order] if diff_order < len(stationarity_results) else None
            },
            'residual_diagnostics': {
                'ljung_box_p_value': ljung_box['lb_pvalue'].iloc[-1],  # Last lag p-value
                'residuals_autocorrelated': ljung_box['lb_pvalue'].iloc[-1] < 0.05,
                'residual_std': residuals.std(),
                'residual_mean': residuals.mean()
            },
            'forecast_validation': {
                'train_size': train_size,
                'test_size': len(test_data),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'last_actual': float(test_data.iloc[-1]),
                'last_forecast': float(forecast.iloc[-1])
            },
            'all_models_tried': [(r['order'], r['aic']) for r in model_results[:5]],  # Top 5
            'data_summary': {
                'total_observations': len(prices),
                'date_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
                'price_range': f"${prices.min():.2f} - ${prices.max():.2f}",
                'mean_price': float(prices.mean()),
                'price_volatility': float(prices.std())
            }
        }
        
        return diagnostics
        
    except Exception as e:
        return {'error': f'Diagnostics failed: {str(e)}'}

# NEW: Compare ARIMA vs actual prices
def compare_arima_vs_actual(ticker, start_date, end_date, forecast_horizon=5):
    """
    Compare ARIMA predictions vs actual prices over a date range.
    Useful for backtesting ARIMA model performance.
    """
    
    if not ARIMA_AVAILABLE:
        return {'error': 'ARIMA functionality not available'}
    
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate date range for comparison
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='D')
        
        results = []
        
        for eval_date in date_range:
            # Get ARIMA prediction
            arima_result = predict_spot_price_arima(ticker, eval_date, forecast_days=1)
            
            # Get actual price
            actual_price = get_spot_price_yahoo(ticker, eval_date)
            
            if ('error' not in arima_result and 
                pd.notna(actual_price) and 
                'predicted_price' in arima_result):
                
                predicted = arima_result['predicted_price']
                error = actual_price - predicted
                error_pct = (error / actual_price) * 100 if actual_price != 0 else 0
                
                results.append({
                    'date': eval_date.strftime('%Y-%m-%d'),
                    'actual_price': round(actual_price, 2),
                    'predicted_price': round(predicted, 2),
                    'error': round(error, 2),
                    'error_percentage': round(error_pct, 2),
                    'confidence_interval': arima_result.get('confidence_interval', {}),
                    'model_type': arima_result.get('model_info', {}).get('model_type', 'ARIMA')
                })
        
        if not results:
            return {'error': 'No valid predictions generated'}
        
        # Calculate summary statistics
        errors = [r['error'] for r in results]
        error_pcts = [r['error_percentage'] for r in results]
        
        summary = {
            'total_predictions': len(results),
            'mean_absolute_error': round(np.mean(np.abs(errors)), 2),
            'root_mean_squared_error': round(np.sqrt(np.mean(np.square(errors))), 2),
            'mean_absolute_percentage_error': round(np.mean(np.abs(error_pcts)), 2),
            'prediction_accuracy': round(100 - np.mean(np.abs(error_pcts)), 2),
            'max_error': round(max(errors, key=abs), 2),
            'min_error': round(min(errors, key=abs), 2)
        }
        
        return {
            'summary': summary,
            'detailed_results': results,
            'ticker': ticker,
            'comparison_period': f"{start_date.date()} to {end_date.date()}"
        }
        
    except Exception as e:
        return {'error': f'Comparison failed: {str(e)}'}

# Example usage functions
def example_arima_prediction():
    """Example of how to use ARIMA spot price prediction"""
    print("=== ARIMA SPOT PRICE PREDICTION EXAMPLE ===")
    
    ticker = "AAPL"
    evaluation_date = "2024-12-01"  # Example date
    
    # Simple ARIMA prediction
    print(f"\n1. Simple ARIMA prediction for {ticker} on {evaluation_date}:")
    result = predict_spot_price_arima(ticker, evaluation_date, use_garch=False)
    
    if 'error' not in result:
        print(f"   Predicted Price: ${result['predicted_price']}")
        print(f"   Confidence Interval: [${result['confidence_interval']['lower']}, ${result['confidence_interval']['upper']}]")
        print(f"   Last Known Price: ${result['last_known_price']}")
        print(f"   Model: {result['model_info']['model_type']} {result['model_info']['order']}")
    else:
        print(f"   Error: {result['error']}")
    
    # ARIMA + GARCH prediction
    print(f"\n2. ARIMA+GARCH prediction for {ticker} on {evaluation_date}:")
    result_garch = predict_spot_price_arima(ticker, evaluation_date, use_garch=True)
    
    if 'error' not in result_garch:
        print(f"   Predicted Price: ${result_garch['predicted_price']}")
        print(f"   Confidence Interval: [${result_garch['confidence_interval']['lower']}, ${result_garch['confidence_interval']['upper']}]")
        print(f"   Model: {result_garch['model_info']['model_type']}")
    else:
        print(f"   Error: {result_garch['error']}")

def example_options_pricing_with_arima():
    """Example of options pricing using ARIMA spot prediction"""
    print("=== OPTIONS PRICING WITH ARIMA EXAMPLE ===")
    
    # First, train the model
    ticker = "AAPL"
    print(f"\nTraining options pricing model for {ticker}...")
    
    # Train with regular Yahoo Finance data
    model, scaler, features, _, _ = run_pipeline(ticker, predict_spot=False)
    
    if model is None:
        print("Failed to train model")
        return
    
    # Test prediction with ARIMA spot price
    print(f"\nTesting option pricing with ARIMA spot prediction...")
    
    result = test_model_prediction(
        model=model,
        scaler=scaler, 
        features=features,
        stock_ticker=ticker,
        strike_price=150,
        option_type='call',
        expiry_date='2024-12-20',
        evaluation_date='2024-12-01',
        predict_spot=True,  # Use ARIMA prediction
        use_garch=False
    )
    
    if 'error' not in result:
        print(f"   ML Model Price: ${result['predictions']['ml_model_price']}")
        print(f"   Black-Scholes Price: ${result['predictions']['black_scholes_price']}")
        print(f"   Spot Price (ARIMA): ${result['market_data']['spot_price']}")
        print(f"   Spot Price Method: {result['market_data']['spot_price_method']}")
        
        if 'spot_price_details' in result:
            details = result['spot_price_details']
            if 'confidence_interval' in details:
                ci = details['confidence_interval']
                print(f"   Spot Price CI: [${ci['lower']}, ${ci['upper']}]")
    else:
        print(f"   Error: {result['error']}")
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error, r2_score

def backtest_model_on_real_data(model, scaler, features, ticker='AAPL', num_cases=50, 
                                predict_spot=True, use_garch=False):
    """
    Backtest the trained ML model against real option prices.
    """
    import random
    from datetime import timedelta
    import pandas as pd

    print(f"\n=== BACKTESTING ON {num_cases} REAL CASES FOR {ticker} ===")
    
    df = get_option_data(ticker, max_expiries=5)
    if df.empty:
        print("❌ No options data available for backtesting.")
        return pd.DataFrame()

    df = prepare_features(df)
    if df.empty:
        print("❌ No valid options after feature preparation.")
        return pd.DataFrame()

    df = df.sample(n=min(num_cases, len(df)), random_state=42).copy()

    results = []

    for i, row in enumerate(df.itertuples(), 1):
        print(f"\n▶️ Test Case {i}/{len(df)} | Date: {row.lastTradeDate.date()}, Expiry: {row.expiration_date.date()}, Strike: {row.strike}, Type: {row.option_type}")

        try:
            result = test_model_prediction(
                model=model,
                scaler=scaler,
                features=features,
                stock_ticker=row.stock,
                strike_price=row.strike,
                option_type=row.option_type,
                expiry_date=row.expiration_date,
                evaluation_date=row.lastTradeDate,
                predict_spot=predict_spot,
                use_garch=use_garch
            )

            if 'error' in result:
                print(f"⚠️ Skipped: {result['error']}")
                continue

            results.append({
                'evaluation_date': row.lastTradeDate,
                'expiry_date': row.expiration_date,
                'option_type': row.option_type,
                'strike_price': row.strike,
                'spot_price': result['market_data']['spot_price'],
                'actual_price': row.lastPrice,
                'ml_price': result['predictions']['ml_model_price'],
                'bs_price': result['predictions']['black_scholes_price'],
                'ml_error_pct': round(abs(result['predictions']['ml_model_price'] - row.lastPrice) / row.lastPrice * 100, 2),
                'bs_error_pct': round(abs(result['predictions']['black_scholes_price'] - row.lastPrice) / row.lastPrice * 100, 2),
            })

        except Exception as e:
            print(f"❌ Exception during test case {i}: {e}")
            continue

    if not results:
        print("❌ No successful predictions.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)

    # R² and MAE calculation
    r2_ml = r2_score(df_results['actual_price'], df_results['ml_price'])
    r2_bs = r2_score(df_results['actual_price'], df_results['bs_price'])

    mae_ml = mean_absolute_error(df_results['actual_price'], df_results['ml_price'])
    mae_bs = mean_absolute_error(df_results['actual_price'], df_results['bs_price'])

    print("\n=== 📊 BACKTEST SUMMARY ===")
    print(f"✅ Test cases: {len(df_results)}")
    print(f"📉 MAE (ML): ${mae_ml:.4f}")
    print(f"📉 MAE (BS): ${mae_bs:.4f}")
    print(f"📊 R² (ML model): {r2_ml:.4f}")
    print(f"📊 R² (Black-Scholes): {r2_bs:.4f}")

    return df_results


    return df_results

if __name__ == "__main__":
    ticker = "AAPL"
    
    # Train your pipeline
    model, scaler, features, _, _ = run_pipeline(ticker, max_expiries=3)

    if model is not None:
        print("✅ Model training successful.")
        
        # Run backtest
        df_backtest = backtest_model_on_real_data(
            model=model,
            scaler=scaler,
            features=features,
            ticker='AAPL',
            num_cases=50,
            predict_spot=True,
            use_garch=False
        )


        if not df_backtest.empty:
            df_backtest.to_csv(f"{ticker}_backtest_results.csv", index=False)
            print("✅ Backtest results saved.")
        else:
            print("⚠️ No valid test cases for backtest.")
    else:
        print("❌ Model training failed. Skipping backtest.")

