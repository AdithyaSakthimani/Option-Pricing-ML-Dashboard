# Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from arch import arch_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import gridfs
import pickle
from bson.objectid import ObjectId
app = Flask(__name__)
CORS(app)
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
client = MongoClient("mongodb+srv://MidnightGamer:Tester123@cluster0.wqmrn.mongodb.net/ChatSpace?retryWrites=true&w=majority&appName=Cluster0")
db = client['option_models']
fs = gridfs.GridFS(db)
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
def train_model(df, model_type='random_forest'):
    """Train a regression model (Linear, RF, Gradient Boosting, XGBoost)"""
    print(f"=== TRAINING MODEL: {model_type.upper()} ===")
    
    if len(df) < 50:
        print("ERROR: Insufficient training data. Need at least 50 records.")
        return None, None, None, None, None
    
    base_features = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'strike', 'spot_price', 'time_to_expiry']
    engineered_features = ['moneyness', 'log_moneyness', 'vol_time', 'volatility', 'intrinsic_value']
    features = base_features + engineered_features

    X = df[features].copy()
    y = df['lastPrice'].copy()
    
    # Remove any remaining NaNs
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n=== TOP FEATURES ===")
        print(feature_importance.head(10))

    return model, scaler, features, y_pred, y_test, feature_importance

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
                         expiry_date, evaluation_date,feature_importance,y_test,y_pred, risk_free_rate=0.05, 
                         predict_spot=False, use_garch=False):
    """Test model prediction with optional ARIMA spot price prediction"""
    
    try:
        # Convert dates
        if feature_importance is None:
            feature_importance = pd.DataFrame(columns=["feature", "importance"])
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
                'strike_price': f"{strike_price:.4e}",
                'option_type': option_type.lower(),
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'evaluation_date': evaluation_date.strftime('%Y-%m-%d'),
                'time_to_expiry_days': (expiry_date - evaluation_date).days,
                'time_to_expiry_years': f"{time_to_expiry:.4e}"
            },
            'market_data': {
                'spot_price': f"{spot_price:.4e}",
                'spot_price_method': 'ARIMA_Prediction' if predict_spot else 'Yahoo_Finance',
                'volatility': f"{volatility:.4e}",
                'volatility_method': vol_method,
                'risk_free_rate': f"{risk_free_rate:.4e}"
            },
            'spot_price_details': spot_price_info,
            'greeks': {k: f"{float(v):.4e}" for k, v in greeks.items()},
            'predictions': {
                'ml_model_price': f"{ml_price:.4e}",
                'black_scholes_price': f"{bs_price:.4e}",
                'price_difference': f"{(ml_price - bs_price):.4e}",
                'percentage_difference': round((ml_price - bs_price) / bs_price * 100, 2) if bs_price > 0 else 0,
                'r2_score': round(r2_score(y_test, y_pred), 4) if y_test is not None and y_pred is not None else 0,
                'mae': round(mean_absolute_error(y_test, y_pred), 4) if y_test is not None and y_pred is not None else 0

            },
            "weightageFactors": [
                {
                    "factor": wf["feature"],
                    "weight": float(wf["importance"]),
                    "color": "#"+''.join(np.random.choice(list('0123456789ABCDEF'), 6))  # Random hex color
                } for wf in feature_importance.to_dict(orient="records")
            ],
            "residualData": [
                {"x": int(i), "residual": float(y_test.iloc[i] - y_pred[i])}
                for i in range(len(y_test))
            ] if y_test is not None and y_pred is not None else []
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# MODIFIED: Main pipeline with ARIMA options
def run_pipeline(ticker, max_expiries=20, predict_spot=False, use_garch=False, model='random_forest'):
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
    model, scaler, features, y_pred, y_test, feature_importance = train_model(df, model_type=model) 
    
    if model is None:
        return None, None, None, None, None
    
    print(f"Pipeline completed successfully for {ticker}")
    return model, scaler, features, y_pred, y_test, feature_importance

# NEW: Utility function for ARIMA model diagnostics
@app.route('/get_model_data', methods=['POST'])
def get_model_data():
    """
    Accepts a POST request with:
    - model: str
    - ticker: str
    - useGarch: bool (optional)
    - predict_spot: bool (optional)
    - expiration_date: str (YYYY-MM-DD)
    - evaluation_date: str (YYYY-MM-DD)
    - option_type: 'call' or 'put'
    - strike_price: float

    Returns model predictions including greeks, volatility, R^2, MAE, etc.
    """
    data = request.get_json()

    model_name = data['model']
    ticker = data['ticker']
    use_garch = data.get('useGarch', False)
    predict_spot = data.get('predict_spot', True)

    # Load model, scaler, features
    model, scaler, features, y_pred, y_test, feature_importance = run_pipeline(ticker, max_expiries=3, model=model_name)

    # Required parameters for prediction
    expiration_date = data['expiration_date']
    evaluation_date = data['evaluation_date']
    option_type = data['option_type']
    strike_price = data['strike_price']

    # Run the prediction
    result = test_model_prediction(
        y_pred=y_pred,
        y_test=y_test,
        feature_importance =feature_importance,
        model=model,
        scaler=scaler,
        features=features,
        stock_ticker=ticker,
        strike_price=strike_price,
        option_type=option_type,
        expiry_date=expiration_date,
        evaluation_date=evaluation_date,
        predict_spot=predict_spot,
        use_garch=use_garch
    )

    return jsonify(result)
import re
def parse_option_contract(contract_name):
    """
    Parse an OCC-style option contract name (e.g., ORCL250711C00125000) into its components.

    Returns:
        dict: {
            'ticker': str,
            'expiration_date': str (YYYY-MM-DD),
            'option_type': 'call' or 'put',
            'strike_price': float
        }

    Raises:
        ValueError: if the format is invalid or components are missing
    """
    try:
        contract_name = contract_name.upper().strip()

        # Find the position of the option type (C or P)
        match = re.match(r"^([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{08})$", contract_name)
        if not match:
            raise ValueError("Invalid contract format - expected format like AAPL240712C00190000")

        ticker = match.group(1)
        year = int(match.group(2)) + 2000
        month = int(match.group(3))
        day = int(match.group(4))
        option_type = 'call' if match.group(5) == 'C' else 'put'
        strike_price = int(match.group(6)) / 1000.0

        # Format date
        try:
            expiration_date = datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid expiration date {year}-{month}-{day}")

        return {
            'ticker': ticker,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike_price': strike_price
        }

    except Exception as e:
        raise ValueError(f"Failed to parse contract name '{contract_name}': {str(e)}")


@app.route('/predict_option_by_contract', methods=['POST'])
def predict_option_by_contract():
    """
    Accepts a POST request with:
    - contract_name: str (e.g., "AAPL240315C00180000")
    - evaluation_date: str (YYYY-MM-DD)
    - model: str (optional, default: 'random_forest')
    - useGarch: bool (optional, default: False)
    - predict_spot: bool (optional, default: True)

    Returns model predictions including greeks, volatility, R^2, MAE, etc.
    """
    try:
        data = request.get_json()
        
        # Required parameters
        contract_name = data.get('contract_name')
        evaluation_date = data.get('evaluation_date')
        
        if not contract_name:
            return jsonify({'error': 'contract_name is required'}), 400
        
        if not evaluation_date:
            return jsonify({'error': 'evaluation_date is required'}), 400
        
        # Optional parameters
        model_name = data.get('model', 'random_forest')
        use_garch = data.get('useGarch', False)
        predict_spot = data.get('predict_spot', True)
        
        # Parse contract name
        try:
            contract_info = parse_option_contract(contract_name)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        ticker = contract_info['ticker']
        expiration_date = contract_info['expiration_date']
        option_type = contract_info['option_type']
        strike_price = contract_info['strike_price']
        # Train model for the ticker
        model, scaler, features, y_pred, y_test, feature_importance = run_pipeline(ticker, max_expiries=3, model=model_name)
        
        if model is None:
            return jsonify({
                'error': f'Failed to train model for {ticker}. Insufficient data or model training failed.'
            }), 500
        
        # Run the prediction
        result = test_model_prediction(
            y_pred=y_pred,
            y_test=y_test,
            feature_importance=feature_importance,
            model=model,
            scaler=scaler,
            features=features,
            stock_ticker=ticker,
            strike_price=strike_price,
            option_type=option_type,
            expiry_date=expiration_date,
            evaluation_date=evaluation_date,
            predict_spot=predict_spot,
            use_garch=use_garch
        )
        
        # Add contract parsing info to result
        if 'error' not in result:
            result['contract_info'] = {
                'contract_name': contract_name,
                'parsed_ticker': ticker,
                'parsed_expiration': expiration_date,
                'parsed_option_type': option_type,
                'parsed_strike': strike_price
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
@app.route('/train_custom_model', methods=['POST'])
def train_custom_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    model_type = request.form.get('model', 'random_forest')
    n_estimators = int(request.form.get('n_estimators', 200))
    max_depth_raw = request.form.get('max_depth')
    max_depth = int(max_depth_raw) if max_depth_raw and max_depth_raw != 'None' else None
    learning_rate = float(request.form.get('learning_rate', 0.1))

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    df = pd.read_csv(file, parse_dates=['expiration_date', 'lastTradeDate'])
    df = prepare_features(df, predict_spot=False)

    if df.empty:
        return jsonify({'error': 'No usable data after preprocessing'}), 400

    base_features = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'strike', 'spot_price', 'time_to_expiry']
    engineered_features = ['moneyness', 'log_moneyness', 'vol_time', 'volatility', 'intrinsic_value']
    features = base_features + engineered_features

    X = df[features].copy()
    y = df['lastPrice'].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=learning_rate, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save to MongoDB GridFS
    model_bin = pickle.dumps(model)
    scaler_bin = pickle.dumps(scaler)
    model_file_id = fs.put(model_bin, filename="model.pkl")
    scaler_file_id = fs.put(scaler_bin, filename="scaler.pkl")

    meta_doc = {
        "model_type": model_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "model_file_id": model_file_id,
        "scaler_file_id": scaler_file_id,
        "created_at": datetime.utcnow()
    }
    model_meta_id = db.models.insert_one(meta_doc).inserted_id

    return jsonify({
        "message": "Model trained and saved to MongoDB",
        "model_id": str(model_meta_id),
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4)
    })
@app.route('/analyze_with_custom_model', methods=['POST'])
def analyze_with_custom_model():
    data = request.get_json()
    model_id = data.get('model_id')

    try:
        # Load model + scaler metadata from MongoDB
        model_doc = db.models.find_one({'_id': ObjectId(model_id)})
        if not model_doc:
            return jsonify({'error': 'Model ID not found'}), 404

        model = pickle.loads(fs.get(model_doc['model_file_id']).read())
        scaler = pickle.loads(fs.get(model_doc['scaler_file_id']).read())

        # Predict using test_model_prediction()
        result = test_model_prediction(
            model=model,
            scaler=scaler,
            features=None,
            stock_ticker=data['ticker'],
            strike_price=data['strike_price'],
            option_type=data['option_type'],
            expiry_date=data['expiration_date'],
            evaluation_date=data['evaluation_date'],
            predict_spot=True,
            use_garch=False,
            y_pred=None,
            y_test=None,
            feature_importance=None
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(
    host='0.0.0.0',  
    port=5000,      
    debug=True       
    )

