import React, { useContext, useEffect } from 'react'
import { TrendingUp, Calculator, BarChart3, Moon, Sun, Activity, Target, Zap, Search, Calendar, Building2, Brain, ArrowRight } from 'lucide-react';
import NoteContext from '../Context/NoteContext';
import '../Styles/ParaInput.css'

function ParaInput() {
    const {darkMode, setDarkMode, companies, models, showResults, setShowResults, loading, setLoading, formData, setFormData, data} = useContext(NoteContext);
    
    // Apply dark mode to document
    useEffect(() => {
        if (darkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    }, [darkMode]);
    
    const handleInputChange = (field, value) => {
        setFormData(prev => ({
            ...prev,
            [field]: value
        }));
    };
    
    const handleAnalyze = async () => {
        if (!formData.companyName || !formData.modelName || !formData.expiryDate) {
            alert('Please fill in all fields');
            return;
        }
        
        setLoading(true);
        // Simulate API call
        setTimeout(() => {
            setLoading(false);
            setShowResults(true);
        }, 2000);
    };

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    return (
        <div className='input-area-main'>


            <div className="header-section">
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
                            <h3  className='f-heading'>Real-Time Option Chain Visualization</h3>
                            <p>
                                Access live option chain data with strike prices, implied volatility, open interest, and real-time pricing, all in a dynamic, user-friendly chart view.
                            </p>
                        </div>

                        <div className="feature-point">
                            <h3  className='f-heading'>Model Customization and Tuning</h3>
                            <p>
                                Fine-tune advanced ML models like XGBoost or Random Forest by configuring hyperparameters such as <code>max_depth</code>, <code>gamma</code>, <code>subsample</code>, and <code>colsample_bytree</code> to fit your trading strategy.
                            </p>
                        </div>
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

                    <div className="form-content">
                        <div className="input-group">
                            <label className="form-label">
                                <Building2 size={18} />
                                Select Company
                            </label>
                            <select
                                className="form-select"
                                value={formData.companyName}
                                onChange={(e) => handleInputChange('companyName', e.target.value)}
                            >
                                <option value="">Choose a company...</option>
                                {companies.map((company) => (
                                    <option key={company.symbol} value={company.symbol}>
                                        {company.symbol} - {company.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="input-group">
                            <label className="form-label">
                                <Brain size={18} />
                                Select Model
                            </label>
                            <select
                                className="form-select"
                                value={formData.modelName}
                                onChange={(e) => handleInputChange('modelName', e.target.value)}
                            >
                                <option value="">Choose a model...</option>
                                {models.map((model) => (
                                    <option key={model.id} value={model.id}>
                                        {model.name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="input-group-calendar">
                            <label className="form-label">
                                <Calendar size={18}/>
                                Expiry Date
                            </label>
                            <input
                                type="date"
                                className="form-input"
                                value={formData.expiryDate}
                                onChange={(e) => handleInputChange('expiryDate', e.target.value)}
                                min={new Date().toISOString().split('T')[0]}
                            />
                        </div>

                        <button 
                            className="analyze-button"
                            onClick={handleAnalyze}
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
                                    Run Analysis
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