# 📈 DerivIQ - Option Price Prediction Platform

A full-stack machine learning web application for predicting American option prices. This project integrates a **Flask-based ML backend**, a **Node.js API connected to MongoDB**, and a **React frontend** to provide interactive visualizations and persistent prediction storage.

---

## 🧠 Project Highlights

- Predict American-style option prices using ML models (Ridge, XGBoost)
- Compare ML predictions with the traditional Binomial pricing model
- Store predictions and user queries in MongoDB via Node.js API
- React-based dashboard for real-time interaction and data visualization

---

## 🚀 Tech Stack

| Layer         | Tech Used                                         |
|---------------|--------------------------------------------------|
| 👨‍💻 Frontend   | React (`src/trading-app`), Axios, Chart.js         |
| 🧠 ML Backend  | Python, Flask, Scikit-learn, XGBoost              |
| 🔌 API Layer   | Node.js, Express.js                               |
| 🗃️ Database    | MongoDB, Mongoose ODM                             |

---

## 📁 Directory Structure

Option-Pricing-ML-Dashboard/
│
├── backend/ # Flask ML server
│ ├── app.py
│ ├── model/ # Model training and utilities
│ └── requirements.txt
│
├── api/ # Node.js API to store predictions
│ ├── server.js
│ ├── routes/
│ ├── models/
│ └── .env
│
├── frontend/
│ └── src/
│ └── trading-app/ # React frontend
│ ├── App.js
│ ├── components/
│ ├── services/
│ └── ...
│
└── README.md # This file

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/11Anan/Option-Pricing-ML-Dashboard.git
cd Option-Pricing-ML-Dashboard
2. Install Dependencies
Flask ML Backend:

bash
Copy
Edit
cd backend/
pip install -r requirements.txt
Node.js API:

bash
Copy
Edit
cd api/
npm install
React Frontend:

bash
Copy
Edit
cd frontend/src/trading-app/
npm install
3. Environment Configuration
Create the following .env files:

api/.env

env
Copy
Edit
MONGO_URI=mongodb://localhost:27017/option_predictions
PORT=3001
(Optional) backend/.env

env
Copy
Edit
FLASK_ENV=development
PORT=5000
4. Run the Application
Start Flask Backend:

bash
Copy
Edit
cd backend/
python app.py
Start Node.js API:

bash
Copy
Edit
cd api/
npm start
Start React Frontend:

bash
Copy
Edit
cd frontend/src/trading-app/
npm start
📊 Key Features
Upload or enter option parameters via frontend

ML models predict option prices using real market data

Predictions are stored in MongoDB through the Node API

Live visualization of:

 Predicted vs Actual Prices

Prediction errors

Option Greeks sensitivity plots

 ML Model Summary
 Input Features: Option Greeks, moneyness, implied volatility, time to expiry, risk-free rate

 Models Used: Ridge Regression, XGBoost

Metrics: MAE, RMSE

Benchmark: Traditional Binomial Model vs ML Prediction

Data Sources: Yahoo Finance (option chains), FRED (risk-free rate)

Future Work
Add deep learning models (LSTM, attention mechanisms)

Expand to support European and exotic options

Add user login + prediction history

Containerize and deploy on platforms like Render or Vercel

