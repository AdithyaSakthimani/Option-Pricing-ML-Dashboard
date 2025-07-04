import mongoose from 'mongoose';

const PredictionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  inputs: {
    stock_ticker: String,
    strike_price: String,
    option_type: String,
    expiry_date: String,
    evaluation_date: String,
    time_to_expiry_days: Number,
    time_to_expiry_years: String
  },
  market_data: {
    spot_price: String,
    spot_price_method: String,
    volatility: String,
    volatility_method: String,
    risk_free_rate: String
  },
  spot_price_details: mongoose.Schema.Types.Mixed, 
  greeks: {
    Delta: String,
    Gamma: String,
    Theta: String,
    Vega: String,
    Rho: String
  },
  predictions: {
    ml_model_price: String,
    black_scholes_price: String,
    price_difference: String,
    percentage_difference: Number,
    r2_score: Number,
    mae: Number
  },
  weightageFactors: [
    {
      factor: String,
      weight: Number,
      color: String
    }
  ],
  residualData: [
    {
      x: Number,
      residual: Number
    }
  ]
}, { timestamps: true });

const Prediction = mongoose.model('Prediction', PredictionSchema);
export default Prediction;
