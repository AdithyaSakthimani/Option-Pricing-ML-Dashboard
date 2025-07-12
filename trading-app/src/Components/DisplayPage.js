import React, { useState, useEffect, useContext } from 'react';
import { TrendingUp, Calculator, BarChart3, Moon, Sun, Activity, Target, Zap, Search, Calendar, Building2, Brain, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import NoteContext from '../Context/NoteContext';
import { GreekCard, MetricCard } from './MainComponents';
import '../Styles/DisplayPage.css';
import axios from 'axios' ; 
function DisplayPage() {
    const {darkMode, setDarkMode, companies, models, showResults, setShowResults, loading, setLoading, formData, setFormData, data
        ,userId , model} = useContext(NoteContext);
    const [isSaved, setIsSaved] = useState(false);
    const [saveMessage, setSaveMessage] = useState('');

    const resetForm = () => {
        setShowResults(false);
        setFormData({
            companyName: '',
            modelName: '',
            expiryDate: ''
        });
    };

    const r2 = data.predictions?.r2_score || 0;
    const mse = data.predictions?.mae || 0;
    const { username, email, isAuthenticated } = useContext(NoteContext); // Make sure you pass userId (ObjectId) or email from NoteContext

    const handleSaveAnalysis = async () => {
  if (!isAuthenticated) {
    alert("Please log in to save your analysis.");
    return;
  }

  const payload = {
  inputs: {
    stock_ticker: data.inputs?.stock_ticker,
    strike_price: data.inputs?.strike_price,
    option_type: data.inputs?.option_type,
    expiry_date: data.inputs?.expiry_date,
    evaluation_date: data.inputs?.evaluation_date || new Date().toISOString().split('T')[0],
    time_to_expiry_days: data.inputs?.time_to_expiry_days,
    time_to_expiry_years: data.inputs?.time_to_expiry_years
  },
  market_data: {
    spot_price: data.market_data?.spot_price,
    spot_price_method: data.market_data?.spot_price_method,
    volatility: data.market_data?.volatility,
    volatility_method: data.market_data?.volatility_method,
    risk_free_rate: data.market_data?.risk_free_rate
  },
  greeks: {
    Delta: data.greeks?.Delta,
    Gamma: data.greeks?.Gamma,
    Theta: data.greeks?.Theta,
    Vega: data.greeks?.Vega,
    Rho: data.greeks?.Rho
  },
  predictions: {
    ml_model_price: data.predictions?.ml_model_price,
    black_scholes_price: data.predictions?.black_scholes_price,
    price_difference: data.predictions?.price_difference,
    percentage_difference: data.predictions?.percentage_difference,
    r2_score: data.predictions?.r2_score,
    mae: data.predictions?.mae
  },
  weightageFactors: data.weightageFactors || [],
  residualData: data.residualData || []
};


  try {
    const res = await axios.post("https://option-pricing-ml-dashboard-1.onrender.com/save-analysis", {
      userId,
      results: payload
    });

    console.log("✅ Save successful", res.data);
    setIsSaved(true);
    setSaveMessage("✅ Analysis saved successfully!");
  } catch (err) {
    console.error("Save failed:", err);
    alert("An error occurred while saving.");
  }
};


    return (
        <>
            <div className="header-section">
                <div className="disp-header-content">
                    <h1 className="main-title">Analysis Results</h1>
                    <p className="main-subtitle">
                        {data.inputs?.stock_ticker || formData.companyName} • 
                            {model} • 
                        Expires: {data.inputs?.expiry_date ? new Date(data.inputs.expiry_date).toLocaleDateString() : new Date(formData.expiryDate).toLocaleDateString()}
                    </p>
                    <div className="button-group">
                        <button className="back-button" onClick={resetForm}>
                            ← New Analysis
                        </button>
                </div>
                </div>
            </div>

            <div className="main-content">
                                        <div className="button-group">
                        <button
                            className="save-button"
                            onClick={handleSaveAnalysis}
                            disabled={isSaved}
                            style={{
                            backgroundColor: isSaved ? "#ccc" : "#06b6d4",
                            cursor: isSaved ? "not-allowed" : "pointer"
                            }}
                        >
                            {isSaved ? "Analysis Saved" : "Save Analysis"}
                        </button>

                        {saveMessage && (
                            <p className='save-msg'>
                            {saveMessage}
                            </p>
                        )}
                    </div>

                <div className="metrics-grid">
                    <MetricCard 
                        title="Model Accuracy (R²)"
                        value={`${(r2 * 100).toFixed(2)}%`}
                        subtitle="Coefficient of Determination"
                        icon={Target}
                    />
                    <MetricCard 
                        title="Mean Squared Error"
                        value={mse.toFixed(6)}
                        subtitle="Lower is better"
                        icon={Activity}
                    />
                    <MetricCard
                        title="Spot Price"
                        value={`$${parseFloat(data.market_data?.spot_price || 0).toFixed(2)}`}
                        subtitle={`Method: ${data.market_data?.spot_price_method || 'N/A'}`}
                        icon={TrendingUp}
                    />
                    <MetricCard 
                        title="Implied Volatility"
                        value={`${(parseFloat(data.market_data?.volatility || 0) * 100).toFixed(2)}%`}
                        subtitle={`Method: ${data.market_data?.volatility_method || 'N/A'}`}
                        icon={Zap}
                    />
                </div>

                {/* Prediction Results */}
                <div className="section-card">
                    <div className="section-header">
                        <Brain size={24} />
                        <h2>Price Predictions</h2>
                    </div>
                    <div className="predictions-grid">
                        <div className="prediction-card">
                            <h3>ML Model Price</h3>
                            <div className="prediction-value">
                                ${parseFloat(data.predictions?.ml_model_price || 0).toFixed(4)}
                            </div>
                        </div>
                        <div className="prediction-card">
                            <h3>Black-Scholes Price</h3>
                            <div className="prediction-value">
                                ${parseFloat(data.predictions?.black_scholes_price || 0).toFixed(4)}
                            </div>
                        </div>
                        <div className="prediction-card">
                            <h3>Price Difference</h3>
                            <div className="prediction-value">
                                ${parseFloat(data.predictions?.price_difference || 0).toFixed(4)}
                            </div>
                            <div className="prediction-percentage">
                                ({data.predictions?.percentage_difference || 0}%)
                            </div>
                        </div>
                    </div>
                </div>

                {/* Greeks Section */}
                <div className="section-card">
                    <div className="section-header">
                        <Calculator size={24} />
                        <h2>Option Greeks</h2>
                    </div>
                    <div className="greeks-grid">
    <GreekCard
        name="Delta (Δ)" 
        value={data.greeks?.Delta || '0.0000e+00'}
        description="Price sensitivity to underlying"
    />
    <GreekCard 
        name="Gamma (Γ)" 
        value={data.greeks?.Gamma || '0.0000e+00'}
        description="Delta sensitivity to underlying"
    />
    <GreekCard 
        name="Theta (Θ)" 
        value={data.greeks?.Theta || '0.0000e+00'}
        description="Time decay sensitivity"
    />
    <GreekCard 
        name="Vega (ν)" 
        value={data.greeks?.Vega || '0.0000e+00'}
        description="Volatility sensitivity"
    />
    <GreekCard 
        name="Rho (ρ)" 
        value={data.greeks?.Rho || '0.0000e+00'}
        description="Interest rate sensitivity"
    />
</div>
                </div>

                {/* Market Data Info */}
                <div className="section-card">
                    <div className="section-header">
                        <Activity size={24} />
                        <h2>Market Data & Inputs</h2>
                    </div>
                    <div className="market-data-grid">
                        <div className="market-data-item">
                            <label>Stock Ticker</label>
                            <value>{data.inputs?.stock_ticker || 'N/A'}</value>
                        </div>
                        <div className="market-data-item">
                            <label>Strike Price</label>
                            <value>${parseFloat(data.inputs?.strike_price || 0).toFixed(2)}</value>
                        </div>
                        <div className="market-data-item">
                            <label>Option Type</label>
                            <value>{data.inputs?.option_type?.toUpperCase() || 'N/A'}</value>
                        </div>
                        <div className="market-data-item">
                            <label>Time to Expiry</label>
                            <value>{data.inputs?.time_to_expiry_days || 0} days</value>
                        </div>
                        <div className="market-data-item">
                            <label>Risk-Free Rate</label>
                            <value>{(parseFloat(data.market_data?.risk_free_rate || 0) * 100).toFixed(2)}%</value>
                        </div>
                    </div>
                </div>

                

                {/* Residual Plot */}
                {data.residualData && data.residualData.length > 0 && (
                    <div className="section-card">
                        <div className="section-header">
                            <Activity size={24} />
                            <h2>Residual Analysis</h2>
                            <div className="residual-stats">
                                Mean: {(data.residualData.reduce((sum, d) => sum + d.residual, 0) / data.residualData.length).toFixed(6)}
                            </div>
                        </div>
                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height={300}>
                                <ScatterChart data={data.residualData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#333' : '#e0e0e0'} />
                                    <XAxis 
                                        type="number"
                                        dataKey="x"
                                        domain={['dataMin', 'dataMax']}
                                        stroke={darkMode ? '#9ca3af' : '#6b7280'}
                                        label={{ value: 'Observation', position: 'insideBottom', offset: -10 }}
                                    />
                                    <YAxis 
                                        type="number"
                                        dataKey="residual"
                                        stroke={darkMode ? '#9ca3af' : '#6b7280'}
                                        label={{ value: 'Residual', angle: -90, position: 'insideLeft' }}
                                    />
                                    <Tooltip 
                                        contentStyle={{
                                            backgroundColor: darkMode ? '#1f2937' : '#ffffff',
                                            border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
                                            borderRadius: '8px',
                                            color: darkMode ? '#ffffff' : '#000000'
                                        }}
                                    />
                                    <Scatter fill="#06b6d4" />
                                    <Line 
                                        type="monotone" 
                                        dataKey={() => 0} 
                                        stroke="#ef4444" 
                                        strokeWidth={2}
                                        strokeDasharray="5 5"
                                    />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}

export default DisplayPage;