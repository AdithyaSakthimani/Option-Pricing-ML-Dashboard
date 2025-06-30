import mongoose from 'mongoose';
const PredictionSchema = new mongoose.Schema({
  stock: {
    type: String,
    required: true
  },
  model: {
    type: String,
    required: true
  },
  formData: {
    bidPrice: {
      type: Number,
      required: true
    },
    askPrice: {
      type: Number,
      required: true
    },
    strikePrice: {
      type: Number,
      required: true
    },
    expiryDate: {
      type: Date,
      required: true
    },
    impliedVolatility: {
      type: Number,
      required: true
    },
    riskFreeRate: {
      type: Number,
      required: true
    },
    timeToMaturity: {
      type: Number,
      required: true
    }
  },
  prediction: {
    callPrice: {
      type: Number,
      required: true
    },
    putPrice: {
      type: Number,
      required: true
    },
    delta: {
      type: Number,
      required: true
    },
    gamma: {
      type: Number,
      required: true
    },
    theta: {
      type: Number,
      required: true
    },
    vega: {
      type: Number,
      required: true
    }
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  }
}, { timestamps: true });
const Prediction_Data_Schema = mongoose.model('Prediction', PredictionSchema);
export default Prediction_Data_Schema;