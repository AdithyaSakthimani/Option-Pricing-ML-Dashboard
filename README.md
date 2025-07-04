# 📈 DerivIQ - Option Price Prediction Platform

DerivIQ is a full-stack machine learning web application for predicting American option prices. It integrates a **Flask-based ML backend**, a **Node.js API with MongoDB** for storing predictions, and a **React frontend** for interactive user input and data visualization.

---

## 🧠 Project Highlights

- Predict American-style option prices using ML models like Ridge Regression and XGBoost
- 🔁 Compare ML predictions with traditional Binomial pricing models
- 🧾 Store user queries and model predictions in MongoDB via a Node.js API
- 📊 View real-time results and analytics in a React-based dashboard

---

## 🚀 Tech Stack

| Layer         | Technologies                                     |
|---------------|--------------------------------------------------|
| 🖥️ Frontend   | React (`src/trading-app`), Axios, Chart.js        |
| 🧠 ML Backend  | Python, Flask, Scikit-learn, XGBoost              |
| 🔌 API Layer   | Node.js, Express.js                               |
| 🗃️ Database    | MongoDB (via Mongoose ODM)                        |

---

## 📁 Directory Structure

```

Option-Pricing-ML-Dashboard/
│
├── backend/                      # Flask ML backend
│   ├── app.py
│   ├── model/                    # Model training & utils
│   └── requirements.txt
│
├── api/                          # Node.js API for DB operations
│   ├── server.js
│   ├── routes/
│   ├── models/
│   └── .env
│
├── frontend/
│   └── src/
│       └── trading-app/          # React frontend
│           ├── App.js
│           ├── components/
│           ├── services/
│           └── ...
│
└── README.md                     # This file

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/11Anan/Option-Pricing-ML-Dashboard.git
cd Option-Pricing-ML-Dashboard
````

---

### 2. Install Dependencies

**Flask ML Backend:**

```bash
cd backend/
pip install -r requirements.txt
```

**Node.js API:**

```bash
cd api/
npm install
```

**React Frontend:**

```bash
cd frontend/src/trading-app/
npm install
```

---


### 3. Run the Application

**Start Flask ML Backend:**

```bash
cd backend/
python app.py
```

**Start Node.js API Server:**

```bash
cd api/
npm start
```

**Start React Frontend:**

```bash
cd frontend/src/trading-app/
npm start
```

---

## 📊 Key Features

* Upload or manually input option contract data
* Predict option prices using Ridge/XGBoost ML models
* Store and retrieve prediction history from MongoDB
* Visualize:

  * ✅ Predicted vs Actual Prices
  * ⚠️ Prediction Errors
  * ⚙️ Greek Sensitivities

---

## 🧬 ML Model Summary

* **Input Features**: Greeks (Delta, Gamma, Vega, Rho), implied volatility, moneyness, days to expiry, risk-free rate
* **Models Used**: Ridge Regression, XGBoost Regressor
* **Evaluation Metrics**: MAE, RMSE
* **Benchmark**: Traditional Binomial Tree Pricing Model
* **Data Sources**: Yahoo Finance (option chains), FRED (risk-free rates)

---

## 🔮 Future Work

* Add support for LSTM/Transformer models
* Extend to European and exotic options
* Implement user login and profile-based prediction history
* Deploy using Render or Vercel

```
