from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pandas_datareader import data as web
import warnings
from flask_cors import CORS
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
def fetch_options_data(stock):
    ticker = yf.Ticker(stock)
    try:
        spot_price = ticker.info.get("regularMarketPrice")
        if spot_price is None or np.isnan(spot_price):
            return pd.DataFrame()
        expiration_dates = ticker.options
    except:
        return pd.DataFrame()

    all_data = []
    for exp_date in expiration_dates:
        try:
            chain = ticker.option_chain(exp_date)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            for df, option_type in [(calls, 'call'), (puts, 'put')]:
                df['option_type'] = option_type
                df['expiration_date'] = exp_date
                df['stock'] = stock
                df['spot_price'] = spot_price
                all_data.append(df)
        except:
            continue
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return {'Delta': np.nan, 'Gamma': np.nan, 'Vega': np.nan, 'Theta': np.nan, 'Rho': np.nan}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    theta = theta_call if option_type == 'call' else theta_put
    rho = rho_call if option_type == 'call' else rho_put
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

@app.route('/predict_option_price', methods=['POST'])
def predict_option_price():
    try:
        data = request.get_json()
        stock = data['stock']
        strike = float(data['strike'])
        expiry = pd.to_datetime(data['expiry_date'])
        option_type = data['option_type']
        model_name = data.get('model', 'XGBoost')

        raw_data = fetch_options_data(stock)
        if raw_data.empty:
            return jsonify({'error': 'No options data fetched'}), 400

        raw_data['lastTradeDate'] = pd.to_datetime(raw_data['lastTradeDate']).dt.date
        raw_data['expiration_date'] = pd.to_datetime(raw_data['expiration_date'])
        raw_data['stock'] = raw_data['stock'].astype(str)
        raw_data = raw_data.fillna(0)

        start = raw_data['expiration_date'].min() - timedelta(days=7)
        end = raw_data['expiration_date'].max() + timedelta(days=7)
        rfr_data = web.DataReader("DTB3", "fred", start, end).ffill()

        raw_data['risk_free_rate'] = raw_data['expiration_date'].map(lambda d: rfr_data.asof(pd.Timestamp(d)).values[0] / 100)
        raw_data['Spot_Price'] = raw_data['spot_price']
        raw_data['time_to_expiry'] = (raw_data['expiration_date'] - datetime.today()).dt.days / 365
        raw_data['moneyness'] = raw_data['Spot_Price'] / raw_data['strike']

        greeks_df = raw_data.apply(lambda row: pd.Series(
            calculate_greeks(row['Spot_Price'], row['strike'], row['time_to_expiry'], row['risk_free_rate'],
                             row['impliedVolatility'], row['option_type'])), axis=1)

        df = pd.concat([raw_data, greeks_df], axis=1)
        df = pd.get_dummies(df, columns=['option_type'])
        df = df.dropna()

        FEATURE_COLUMNS = ["strike", "impliedVolatility", "time_to_expiry", "moneyness",
                           "option_type_call", "option_type_put", "risk_free_rate",
                           "Delta", "Gamma", "Vega", "Theta", "Rho"]

        X = df[FEATURE_COLUMNS]
        y = df['lastPrice']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            "xgboost": XGBRegressor(n_estimators=100, random_state=42),
            "linear-regression": LinearRegression(),
            "random-forest": RandomForestRegressor(),
            "gradient-boosting": GradientBoostingRegressor()
        }

        for model in models.values():
            model.fit(X_scaled, y)

        selected_model = models[model_name]
        y_pred = selected_model.predict(X_scaled)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        n, p = X.shape
        adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

        residualData = [{"x": i + 1, "residual": round(float(pred) - float(actual), 4)}
                        for i, (pred, actual) in enumerate(zip(y_pred, y))]

        # Unified importance extraction for all models
        if hasattr(selected_model, 'feature_importances_'):
            importances = selected_model.feature_importances_
        elif hasattr(selected_model, 'coef_'):
            importances = np.abs(selected_model.coef_)
        else:
            importances = None

        color_map = {
            'impliedVolatility': '#8b5cf6',
            'time_to_expiry': '#06b6d4',
            'strike': '#10b981',
            'risk_free_rate': '#f59e0b',
            'dividendYield': '#ef4444'
        }
        readable_map = {
            "impliedVolatility": "Implied Volatility",
            "time_to_expiry": "Time to Expiry",
            "strike": "Strike Price",
            "risk_free_rate": "Interest Rate",
            "dividendYield": "Dividend Yield"
        }

        weightageFactors = []
        if importances is not None:
            total = np.sum(importances)
            for name, score in zip(FEATURE_COLUMNS, importances):
                if name in readable_map:
                    weightageFactors.append({
                        "factor": readable_map[name],
                        "weight": round((score / total) * 100, 1),
                        "color": color_map.get(name, '#888888')
                    })

        ticker = yf.Ticker(stock)
        spot = ticker.info.get('regularMarketPrice')
        if not spot:
            return jsonify({'error': 'Spot price unavailable'}), 400

        r = rfr_data.asof(pd.Timestamp(expiry)).values[0] / 100
        T = (expiry - datetime.today()).days / 365
        moneyness = spot / strike
        sigma = 0.3

        greeks = calculate_greeks(spot, strike, T, r, sigma, option_type)

        input_df = pd.DataFrame([{**{
            "strike": strike,
            "impliedVolatility": sigma,
            "time_to_expiry": T,
            "moneyness": moneyness,
            "option_type_call": 1 if option_type == 'call' else 0,
            "option_type_put": 1 if option_type == 'put' else 0,
            "risk_free_rate": r
        }, **greeks}])

        input_df = input_df[FEATURE_COLUMNS]
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)[0]

        return jsonify({
            "lastPrice": round(float(prediction), 2),
            "mse": round(float(mse), 4),
            "r2": round(float(r2), 4),
            "adjusted_r2": round(float(adjusted_r2), 4),
            "impliedVolatility": round(float(sigma), 4),  # <-- ✅ Added here
            "greeks": {
                "delta": round(greeks.get("Delta", 0), 4),
                "gamma": round(greeks.get("Gamma", 0), 4),
                "theta": round(greeks.get("Theta", 0), 4),
                "vega": round(greeks.get("Vega", 0), 4),
                "rho": round(greeks.get("Rho", 0), 4),
            },
            "weightageFactors": weightageFactors,
            "residualData": residualData
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
