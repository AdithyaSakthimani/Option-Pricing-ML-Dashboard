import React, { useState, useEffect, useContext } from 'react';
import { Trash2, Eye, Calendar, TrendingUp, BarChart3, DollarSign, Clock, AlertCircle } from 'lucide-react';
import '../Styles/PastAnalysis.css';
import NoteContext from '../Context/NoteContext';
import axios from 'axios';
const PastAnalysis = () => {
  const { darkMode,analyses,setAnalyses,userId } = useContext(NoteContext);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [sortBy, setSortBy] = useState('date');
  const [filterType, setFilterType] = useState('all');
 useEffect(() => {
  const fetchAnalyses = async () => {
    try {
      const res = await axios.post("http://localhost:8000/get-user-analyses", {
        userId 
      });
      console.log(res.data); 
      setAnalyses(res.data.data) ; 
    } catch (err) {
      console.error("Error fetching analyses:", err);
    }
  };

  fetchAnalyses();
}, []);

  // Function to add new analysis (to be called from parent component)
  const addAnalysis = (analysisData) => {
    const newAnalysis = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...analysisData
    };
    setAnalyses(prev => [newAnalysis, ...prev]);
  };

  // Remove analysis
  const removeAnalysis = async (id) => {
  try {
    await axios.delete(`http://localhost:8000/delete-analysis/${id}`, {
      data: { userId }
    });
    setAnalyses(prev => prev.filter(analysis => analysis._id !== id));
  } catch (error) {
    console.error("Error deleting analysis:", error);
    alert("Failed to delete analysis. Please try again.");
  }
};

  // View analysis details
  const viewAnalysis = (analysis) => {
    setSelectedAnalysis(analysis);
    setShowModal(true);
  };

  // Sort analyses
  const sortedAnalyses = [...analyses].sort((a, b) => {
    switch (sortBy) {
      case 'date':
        return new Date(b.timestamp) - new Date(a.timestamp);
      case 'ticker':
        return a.inputs.stock_ticker.localeCompare(b.inputs.stock_ticker);
      case 'strike':
        return parseFloat(a.inputs.strike_price) - parseFloat(b.inputs.strike_price);
      case 'accuracy':
        return b.predictions.r2_score - a.predictions.r2_score;
      default:
        return 0;
    }
  });

  // Filter analyses
  const filteredAnalyses = sortedAnalyses.filter(analysis => {
    if (filterType === 'all') return true;
    return analysis.inputs.option_type === filterType;
  });

  const formatNumber = (value) => {
    const num = parseFloat(value);
    return isNaN(num) ? value : num.toFixed(4);
  };

  const formatCurrency = (value) => {
    const num = parseFloat(value);
    return isNaN(num) ? value : `$${num.toFixed(2)}`;
  };

  const formatPercentage = (value) => {
    return `${value}%`;
  };

  return (
    <div className={`past-analysis-container ${darkMode ? 'dark' : 'light'}`}>
      <div className="analysis-header">
        <h2>Past Option Analyses</h2>
        <div className="analysis-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="sort-select"
          >
            <option value="date">Sort by Date</option>
            <option value="ticker">Sort by Ticker</option>
            <option value="strike">Sort by Strike Price</option>
            <option value="accuracy">Sort by Accuracy</option>
          </select>
          
          <select 
            value={filterType} 
            onChange={(e) => setFilterType(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Types</option>
            <option value="call">Call Options</option>
            <option value="put">Put Options</option>
          </select>
        </div>
      </div>

      {analyses.length === 0 ? (
        <div className="no-analyses">
          <AlertCircle size={48} />
          <p>No past analyses found</p>
          <span>Your option analyses will appear here once you start analyzing options</span>
        </div>
      ) : (
        <div className="analyses-grid">
          {filteredAnalyses.map((analysis) => (
            <div key={analysis._id} className={`analysis-card ${darkMode ? 'dark' : 'light'}`}>
              <div className="card-header">
                <div className="ticker-info">
                  <h3 className={`analysis-hed ${darkMode ? 'dark' : 'light'}`}>{analysis.inputs.stock_ticker}</h3>
                  <span className={`option-type ${analysis.inputs.option_type}`}>
                    {analysis.inputs.option_type.toUpperCase()}
                  </span>
                </div>
                <div className="card-actions">
                  <button 
                    onClick={() => viewAnalysis(analysis)}
                    className="action-btn view-btn"
                    title="View Details"
                  >
                    <Eye size={16} />
                  </button>
                  <button 
                      onClick={() => removeAnalysis(analysis._id)}
                      className="action-btn delete-btn"
                      title="Delete Analysis"
                    >
                      <Trash2 size={16} />
                    </button>

                </div>
              </div>

              <div className="card-content">
                <div className="info-row">
                  <span className="label">Strike Price:</span>
                  <span className="value">{formatCurrency(analysis.inputs.strike_price)}</span>
                </div>
                <div className="info-row">
                  <span className="label">Spot Price:</span>
                  <span className="value">{formatCurrency(analysis.market_data.spot_price)}</span>
                </div>
                <div className="info-row">
                  <span className="label">Expiry:</span>
                  <span className="value">{analysis.inputs.expiry_date}</span>
                </div>
                <div className="info-row">
                  <span className="label">ML Price:</span>
                  <span className="value">{formatCurrency(analysis.predictions.ml_model_price)}</span>
                </div>
                <div className="info-row">
                  <span className="label">BS Price:</span>
                  <span className="value">{formatCurrency(analysis.predictions.black_scholes_price)}</span>
                </div>
                <div className="info-row">
                  <span className="label">Accuracy (R²):</span>
                  <span className="value">{analysis.predictions.r2_score}</span>
                </div>
              </div>

              <div className="card-footer">
                <div className="timestamp">
                  <Clock size={14} />
                  {new Date(analysis.timestamp).toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal for detailed view */}
      {showModal && selectedAnalysis && (
        <div className={`modal-overlay ${darkMode ? 'dark' : 'light'}`} onClick={() => setShowModal(false)}>
          <div className={`modal-content ${darkMode ? 'dark' : 'light'}`} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedAnalysis.inputs.stock_ticker} - Detailed Analysis</h2>
              <button 
                onClick={() => setShowModal(false)}
                className="close-btn"
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              <div className="detail-section">
                <h3><DollarSign size={20} /> Input Parameters</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="detail-label">Stock Ticker:</span>
                    <span className="detail-value">{selectedAnalysis.inputs.stock_ticker}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Strike Price:</span>
                    <span className="detail-value">{formatCurrency(selectedAnalysis.inputs.strike_price)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Option Type:</span>
                    <span className="detail-value">{selectedAnalysis.inputs.option_type.toUpperCase()}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Expiry Date:</span>
                    <span className="detail-value">{selectedAnalysis.inputs.expiry_date}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Time to Expiry:</span>
                    <span className="detail-value">{selectedAnalysis.inputs.time_to_expiry_days} days</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3><BarChart3 size={20} /> Market Data</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="detail-label">Spot Price:</span>
                    <span className="detail-value">{formatCurrency(selectedAnalysis.market_data.spot_price)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Volatility:</span>
                    <span className="detail-value">{formatNumber(selectedAnalysis.market_data.volatility)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Risk-Free Rate:</span>
                    <span className="detail-value">{formatNumber(selectedAnalysis.market_data.risk_free_rate)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Spot Price Method:</span>
                    <span className="detail-value">{selectedAnalysis.market_data.spot_price_method}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Volatility Method:</span>
                    <span className="detail-value">{selectedAnalysis.market_data.volatility_method}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3><TrendingUp size={20} /> Predictions & Performance</h3>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="detail-label">ML Model Price:</span>
                    <span className="detail-value">{formatCurrency(selectedAnalysis.predictions.ml_model_price)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Black-Scholes Price:</span>
                    <span className="detail-value">{formatCurrency(selectedAnalysis.predictions.black_scholes_price)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Price Difference:</span>
                    <span className="detail-value">{formatCurrency(selectedAnalysis.predictions.price_difference)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Percentage Difference:</span>
                    <span className="detail-value">{formatPercentage(selectedAnalysis.predictions.percentage_difference)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">R² Score:</span>
                    <span className="detail-value">{selectedAnalysis.predictions.r2_score}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">MAE:</span>
                    <span className="detail-value">{selectedAnalysis.predictions.mae}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h3><BarChart3 size={20} /> Greeks</h3>
                <div className="detail-grid">
                  {Object.entries(selectedAnalysis.greeks).map(([key, value]) => (
                    <div key={key} className="detail-item">
                      <span className="detail-label">{key.charAt(0).toUpperCase() + key.slice(1)}:</span>
                      <span className="detail-value">{formatNumber(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PastAnalysis;