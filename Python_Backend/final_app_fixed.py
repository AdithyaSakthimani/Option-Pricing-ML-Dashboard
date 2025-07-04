# Final modified code to limit test cases and fix repeated ARIMA calls

# -- Original imports and functions retained from user's full script above --

# Update in run_pipeline(): limit to 30 cases after feature preparation
def run_pipeline(ticker, max_expiries=20, predict_spot=False, use_garch=False, model='random_forest'):
    print(f"=== RUNNING PIPELINE FOR {ticker} ===")
    print(f"Spot price prediction: {'ARIMA' + ('+GARCH' if use_garch else '') if predict_spot else 'Yahoo Finance'}")
    
    df = get_option_data(ticker, max_expiries)
    if df.empty:
        print(f"No options data found for {ticker}")
        return None, None, None, None, None
    
    df = prepare_features(df, predict_spot=predict_spot, use_garch=use_garch)
    if df.empty or len(df) < 50:
        print(f"Insufficient quality data for {ticker}")
        return None, None, None, None, None

    df = df.sample(n=min(30, len(df)), random_state=42)  # LIMIT to 30 test cases

    model, scaler, features, y_pred, y_test, feature_importance = train_model(df, model_type=model)
    if model is None:
        return None, None, None, None, None
    
    print(f"Pipeline completed successfully for {ticker}")
    return model, scaler, features, y_pred, y_test, feature_importance

# Ensure the app is only served when not imported
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
