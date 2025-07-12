import React, { useContext, useEffect, useState } from 'react'
import { TrendingUp, Calculator, BarChart3, Moon, Sun, Activity, Target, Zap, Search, Calendar, Building2, Brain, ArrowRight, DollarSign, ToggleLeft, ToggleRight, Settings, TrendingDown } from 'lucide-react';
import NoteContext from '../Context/NoteContext';
import '../Styles/ParaInput.css'
import axios from 'axios';
import TickerInput from './TickerSuggest';

function ParaInput() {
    const {darkMode, setDarkMode, companies, models, showResults, setShowResults, loading, setLoading, formData, setFormData, data
        ,option_types, setData, model , setModel 
    } = useContext(NoteContext);
    
    const [isQuickMode, setIsQuickMode] = useState(false);
    
    // Apply dark mode to document
    useEffect(() => {
        if (darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    }, [darkMode]);
    
    // Initialize form data with new fields
    useEffect(() => {
        if (!formData.evaluation_date) {
            const today = new Date().toISOString().split('T')[0];
            setFormData(prev => ({
                ...prev,
                evaluation_date: today,
                predict_spot: false,
                useGarch: false
            }));
        }
    }, []);
    
    // Get date limits (today + 7 days)
    const getDateLimits = () => {
        const today = new Date();
        const maxDate = new Date();
        maxDate.setDate(today.getDate() + 7);
        
        return {
            min: today.toISOString().split('T')[0],
            max: maxDate.toISOString().split('T')[0]
        };
    };
    
    const handleInputChange = (field, value) => {
        setFormData(prev => ({
            ...prev,
            [field]: value
        }));
    };
    
    const handleQuickAnalyze = async () => {
        if (!formData.contract) {
            alert('Please enter a contract name.');
            return;
        }

        setLoading(true);
        try {
            console.log('Quick Analysis for contract:', formData.contract);
            const response = await axios.post('http://localhost:5000/predict_option_by_contract', {
                contract_name: formData.contract,
                evaluation_date: formData.evaluation_date,
                model: formData.model || 'random_forest',    
                useGarch: formData.useGarch || false,
                predict_spot: formData.predict_spot || false
            });
            if (response.data.error) {
            alert(response.data.error);
            setLoading(false);
            return;
            }

            console.log('Quick analysis result:', response.data);
            setData(response.data);
            setShowResults(true);
        } catch (error) {
            console.error('Error during quick contract analysis:', error);
            alert('An error occurred while analyzing the contract.');
        } finally {
            setLoading(false);
        }
        };

    
    const handleAnalyze = async () => {
    const { ticker, model, expiration_date, strike_price, option_type, evaluation_date, predict_spot, useGarch } = formData;

    // Basic validation
    if (!ticker || !model || !expiration_date || !strike_price || !option_type || !evaluation_date) {
        alert('Please fill in all required fields: Ticker, Model, Option Type, Strike Price, Evaluation Date, and Expiration Date.');
        return;
    }

    // Validate strike price
    const parsedStrike = parseFloat(strike_price);
    if (isNaN(parsedStrike) || parsedStrike <= 0) {
        alert('Please enter a valid strike price greater than 0.');
        return;
    }

    // Validate expiry date > evaluation date
    const expiry = new Date(expiration_date);
    const evalDate = new Date(evaluation_date);
    if (isNaN(expiry.getTime()) || isNaN(evalDate.getTime())) {
        alert('Invalid date format. Please use a valid Evaluation and Expiration date.');
        return;
    }
    if (expiry <= evalDate) {
        alert('Expiration date must be after the evaluation date.');
        return;
    }

    setLoading(true);

    try {
        const response = await axios.post('http://localhost:5000/get_model_data', {
            ticker: ticker.trim().toUpperCase(),
            model,
            option_type,
            strike_price: parsedStrike,
            expiration_date,
            evaluation_date,
            predict_spot,
            useGarch
        });

        console.log('Analysis result:', response.data);
        setModel(model) ; 
        setData(response.data);
        setShowResults(true);
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('An error occurred during analysis. Please check the inputs or try again later.');
    } finally {
        setLoading(false);
    }
};


    const dateLimits = getDateLimits();

    return (
        <div className='input-area-main'>

                <div className="header-content">
                    <h1 className="main-title">Options Analysis Platform</h1>
                    <p className="main-subtitle">Enter your trading parameters to get comprehensive analysis</p>
                    
                    <div className="features-container">
                        <div className="feature-point">
                            <h3 className='f-heading'>AI-Powered Strategy Suggestions</h3>
                            <p>
                                Automatically recommends optimal trading strategies such as Covered Call, Iron Condor, or Straddle based on the selected model and live market conditions.
                            </p>
                        </div>

                        <div className="feature-point">
                            <h3 className='f-heading'>Real-Time Option Chain Visualization</h3>
                            <p>
                                Access live option chain data with strike prices, implied volatility, open interest, and real-time pricing, all in a dynamic, user-friendly chart view.
                            </p>
                        </div>

                        <div className="feature-point">
                            <h3 className='f-heading'>Model Customization and Tuning</h3>
                            <p>
                                Fine-tune advanced ML models like XGBoost or Random Forest by configuring hyperparameters such as <code>max_depth</code>, <code>gamma</code>, <code>subsample</code>, and <code>colsample_bytree</code> to fit your trading strategy.
                            </p>
                        </div>
                    </div>
            </div>

            <div className="form-container">
                <div className="form-card">
                    <div className="form-header">
                        <Calculator size={28} />
                        <div>
                            <h2>Analysis Parameters</h2>
                            <p>Configure your options analysis settings</p>
                        </div>
                    </div>

                    {/* Mode Toggle */}
                    <div className="mode-toggle-container">
                        <div className="mode-toggle">
                            <span className={`mode-label ${!isQuickMode ? 'active' : ''}`}>Detailed Analysis</span>
                            <button 
                                className="toggle-button"
                                onClick={() => setIsQuickMode(!isQuickMode)}
                            >
                                {isQuickMode ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
                            </button>
                            <span className={`mode-label ${isQuickMode ? 'active' : ''}`}>Quick Ticker Analysis</span>
                        </div>
                        <p className="mode-description">
                            {isQuickMode 
                                ? "Enter just a ticker symbol for quick analysis with default parameters" 
                                : "Configure all parameters for detailed options analysis"
                            }
                        </p>
                    </div>

                    <div className="form-content">
                        {isQuickMode ? (
                            <>
                        {/* Contract Name Input */}
                        <div className="input-group">
                        <label className="form-label">
                            <Zap size={18} />
                            Contract Name (e.g., AAPL240712C00190000)
                        </label>
                        <input
                            type="text"
                            className="form-input-ticker"
                            value={formData.contract || ''}
                            placeholder="Enter OCC-style contract name"
                            onChange={(e) => handleInputChange('contract', e.target.value.toUpperCase())}
                        />
                        </div>

                        {/* Evaluation Date */}
                        <div className="input-group">
                        <label className="form-label">
                            <Calendar size={18} />
                            Evaluation Date
                        </label>
                        <input
                            type="date"
                            className="form-input-calendar"
                            value={formData.evaluation_date || ''}
                            onChange={(e) => handleInputChange('evaluation_date', e.target.value)}
                        />
                        </div>

                        {/* Model Selection */}
                        <div className="input-group">
                        <label className="form-label">
                            <Brain size={18} />
                            Model to Use
                        </label>
                        <select
                            className="form-select"
                            value={formData.model || ''}
                            onChange={(e) => handleInputChange('model', e.target.value)}
                        >
                            <option value="" style={{ color: 'var(--text-secondary)' }}>Choose a model...</option>
                            {models.map((model) => (
                            <option key={model.id} value={model.id}>
                                {model.name}
                            </option>
                            ))}
                        </select>
                        </div>
                    </>
                    ) : (
                    <TickerInput
                        value={formData.ticker || ''}
                        onChange={(val) => handleInputChange('ticker', val.toUpperCase())}
                        isQuickMode={isQuickMode}
                        placeholder="AAPL, MSFT, TSLA, etc."
                    />
                    )}


                        {!isQuickMode && (
                            <>  
                                <div className="input-group">
                                        <label className="form-label">
                                            <Brain size={18} />
                                            Model to Use
                                        </label>
                                        <select
                                            className="form-select"
                                            value={formData.model || ''}
                                            onChange={(e) => handleInputChange('model', e.target.value)}
                                        >
                                            <option value="" style={{ color: 'var(--text-secondary)' }}>Choose a model...</option>
                                            {models.map((model) => (
                                            <option key={model.id} value={model.id}>
                                                {model.name}
                                            </option>
                                            ))}
                                        </select>
                                        </div>
                                {/* Option Type */}
                                <div className="input-group">
                                    <label className="form-label">
                                        <TrendingUp size={18} />
                                        Option Type
                                    </label>
                                    <select
                                        className="form-select"
                                        value={formData.option_type || ''}
                                        onChange={(e) => handleInputChange('option_type', e.target.value)}
                                    >
                                        <option value="" style={{color: 'var(--text-secondary)'}}>Choose option type...</option>
                                        <option value="call">Call Option</option>
                                        <option value="put">Put Option</option>
                                    </select>
                                </div>
                                
                                {/* Strike Price */}
                                <div className="input-group">
                                    <label className="form-label">
                                        <DollarSign size={18} />
                                        Strike Price
                                    </label>
                                    <div className="input-wrapper">
                                        <input
                                            type="number"
                                            className="form-input-with-dollar"
                                            value={formData.strike_price || ''}
                                            placeholder="Enter strike price"
                                            onChange={(e) => handleInputChange('strike_price', e.target.value)}
                                            min={0}
                                            step="0.01"
                                        />
                                    </div>
                                </div>
                                
                                {/* Expiration Date */}
                                <div className="input-group">
                                    <label className="form-label">
                                        <Calendar size={18}/>
                                        Expiration Date
                                    </label>
                                    <input
                                        type="date"
                                        className="form-input-calendar"
                                        value={formData.expiration_date || ''}
                                        onChange={(e) => handleInputChange('expiration_date', e.target.value)}
                                        min={dateLimits.min}
                                        max={dateLimits.max}
                                        placeholder="Select expiry date"
                                    />
                                </div>

                                {/* Evaluation Date */}
                                <div className="input-group">
                                    <label className="form-label">
                                        <Calendar size={18}/>
                                        Evaluation Date
                                    </label>
                                    <input
                                        type="date"
                                        className="form-input-calendar"
                                        value={formData.evaluation_date || ''}
                                        onChange={(e) => handleInputChange('evaluation_date', e.target.value)}
                                        placeholder="Select evaluation date"
                                    />
                                </div>

                                {/* Stock Price Estimation Method */}
                                <div className="input-group">
                                    <label className="form-label">
                                        <Settings size={18} />
                                        Stock Price Estimation Method
                                    </label>
                                    <div className="toggle-option">
                                        <label className="toggle-label">
                                            <input
                                                type="checkbox"
                                                checked={formData.predict_spot || false}
                                                onChange={(e) => handleInputChange('predict_spot', e.target.checked)}
                                            />
                                            <span className="toggle-description">
                                                Predict future spot price (unchecked = use today's price)
                                            </span>
                                        </label>
                                    </div>
                                </div>

                                {/* Advanced Prediction (Volatility Forecasting) */}
                                {formData.predict_spot && (
                                    <div className="input-group advanced-option">
                                        <label className="form-label">
                                            <TrendingDown size={18} />
                                            Use Advanced Prediction (Volatility Forecasting)
                                        </label>
                                        <div className="toggle-option">
                                            <label className="toggle-label">
                                                <input
                                                    type="checkbox"
                                                    checked={formData.useGarch || false}
                                                    onChange={(e) => handleInputChange('useGarch', e.target.checked)}
                                                />
                                                <span className="toggle-description">
                                                    Use GARCH model (unchecked = basic ARIMA)
                                                </span>
                                            </label>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        <button 
                            className="analyze-button"
                            onClick={isQuickMode ? handleQuickAnalyze : handleAnalyze}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <div className="spinner"></div>
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <ArrowRight size={20} />
                                    {isQuickMode ? 'Quick Analysis' : 'Run Analysis'}
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ParaInput