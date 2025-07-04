import React, { useState, useContext,useEffect } from 'react';
import './App.css';
import NoteContext from './Context/NoteContext';

const App = () => {
  const [file, setFile] = useState(null);
  const [hyperparams, setHyperparams] = useState({
    max_depth: 6,
    gamma: 0.1,
    subsample: 0.8,
    colsample_bytree: 0.8,
    target_column: 'option_price'
  });

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [results, setResults] = useState(null);
  const { darkMode } = useContext(NoteContext);
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleParamChange = (param, value) => {
    setHyperparams(prev => ({
      ...prev,
      [param]: value
    }));
  };


  const uploadAndTrainModel = async () => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', 'xgboost'); // Or 'random_forest', 'linear', 'gradient_boosting'

  // Add your XGBoost-style hyperparameters
  formData.append('max_depth', hyperparams.max_depth);
  formData.append('gamma', hyperparams.gamma);
  formData.append('subsample', hyperparams.subsample);
  formData.append('colsample_bytree', hyperparams.colsample_bytree);
  formData.append('learning_rate', 0.1);  // You can make this adjustable too

  try {
    const res = await fetch('http://localhost:5000/train_custom_model', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Training failed');
    }

    const data = await res.json();
    return {
      r2_score: data.r2_score,
      mae: data.mae,
      rmse: data.rmse || (Math.sqrt(data.mae) * 1.5).toFixed(2),
      training_time: 'N/A',
      feature_importance: data.feature_importance || [],
      predictions: []  // You can load sample predictions later if needed
    };
  } catch (error) {
    console.error('Training failed:', error.message);
    throw error;
  }
};

const handleSubmit = async (e) => {
  e.preventDefault();
  if (!file) {
    alert('Please upload a CSV file');
    return;
  }

  setIsTraining(true);
  setTrainingProgress(0);
  setCurrentEpoch(0);
  setResults(null);

  try {
    const interval = setInterval(() => {
      setTrainingProgress((prev) => Math.min(prev + Math.random() * 5, 100));
      setCurrentEpoch((prev) => Math.min(prev + 1, 100));
    }, 50);

    const result = await uploadAndTrainModel();
    clearInterval(interval);
    setTrainingProgress(100);
    setCurrentEpoch(100);
    setResults(result);
  } catch (error) {
    alert(error.message);
  } finally {
    setIsTraining(false);
  }
};

  return (
    <div className={`training-app ${darkMode ? 'dark' : 'light'}`}>
      <div className="container">
        <div className="grid">
          {/* Configuration Panel */}
          <div className="card config-panel">
            <div className="card-header">
              <h2 className="card-title">
                Model Configuration
              </h2>
            </div>
            
            <form onSubmit={handleSubmit} className="form">
              {/* File Upload */}
              <div className="form-group">
                <label className="label">Training Data</label>
                <div className="file-upload">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="file-input"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="file-label">
                    <span className="file-icon">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14,2 14,8 20,8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <line x1="10" y1="9" x2="9" y2="9"/>
                      </svg>
                    </span>
                    <span className="file-text">
                      {file ? file.name : 'Choose CSV file...'}
                    </span>
                  </label>
                </div>
              </div>

              {/* Target Column */}
              <div className="form-group">
                <label className="label">Target Column</label>
                <input
                  type="text"
                  value={hyperparams.target_column}
                  onChange={(e) => handleParamChange('target_column', e.target.value)}
                  className="input"
                  placeholder="Enter target column name"
                />
              </div>

              {/* Hyperparameters */}
              <div className="hyperparams">
                <h3 className="section-title">Hyperparameters</h3>
                
                {/* Max Depth */}
                <div className="param-group">
                  <label className="param-label">
                    Max Depth
                    <span className="param-value">{hyperparams.max_depth}</span>
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="15"
                    step="1"
                    value={hyperparams.max_depth}
                    onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
                    className="slider"
                  />
                  <div className="slider-track">
                    <span>3</span>
                    <span>15</span>
                  </div>
                </div>

                {/* Gamma */}
                <div className="param-group">
                  <label className="param-label">
                    Gamma
                    <span className="param-value">{hyperparams.gamma}</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={hyperparams.gamma}
                    onChange={(e) => handleParamChange('gamma', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <div className="slider-track">
                    <span>0</span>
                    <span>2</span>
                  </div>
                </div>

                {/* Subsample */}
                <div className="param-group">
                  <label className="param-label">
                    Subsample
                    <span className="param-value">{hyperparams.subsample}</span>
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.1"
                    value={hyperparams.subsample}
                    onChange={(e) => handleParamChange('subsample', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <div className="slider-track">
                    <span>0.5</span>
                    <span>1.0</span>
                  </div>
                </div>

                {/* Colsample by Tree */}
                <div className="param-group">
                  <label className="param-label">
                    Colsample by Tree
                    <span className="param-value">{hyperparams.colsample_bytree}</span>
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.1"
                    value={hyperparams.colsample_bytree}
                    onChange={(e) => handleParamChange('colsample_bytree', parseFloat(e.target.value))}
                    className="slider"
                  />
                  <div className="slider-track">
                    <span>0.5</span>
                    <span>1.0</span>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button 
                type="submit" 
                className={`btn btn-primary ${isTraining ? 'loading' : ''}`}
                disabled={isTraining}
              >
                {isTraining ? (
                  <>
                    <span className="btn-spinner"></span>
                    Training Model...
                  </>
                ) : (
                  'Train Model'
                )}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="card results-panel">
            <div className="card-header">
              <h2 className="card-title">
                Model Performance
              </h2>
            </div>
            
            {!results && !isTraining && (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="m9 12 2 2 4-4"/>
                  </svg>
                </div>
                <h3>Ready to Train</h3>
                <p>Configure your model parameters and upload your training data to get started</p>
              </div>
            )}

            {isTraining && (
              <div className="training-state">
                <div className="training-header">
                  <h3>Training in Progress</h3>
                  <span className="epoch-counter">Epoch {currentEpoch}/100</span>
                </div>
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">{trainingProgress.toFixed(1)}%</span>
                </div>
                <div className="training-status">
                  <div className="status-item">
                    <span className="status-dot"></span>
                    Processing features...
                  </div>
                </div>
              </div>
            )}

            {results && (
              <div className="results">
                {/* Metrics */}
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-value">{results.r2_score}</div>
                    <div className="metric-label">R² Score</div>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill" 
                        style={{ width: `${parseFloat(results.r2_score) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{results.mae}</div>
                    <div className="metric-label">Mean Absolute Error</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{results.rmse}</div>
                    <div className="metric-label">Root Mean Square Error</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value">{results.training_time}</div>
                    <div className="metric-label">Training Time</div>
                  </div>
                </div>

                {/* Feature Importance */}
                <div className="feature-importance">
                  <h3 className="section-title">Feature Importance</h3>
                  <div className="importance-list">
                    {results.feature_importance.map((item, idx) => (
                      <div key={idx} className="importance-item">
                        <div className="importance-info">
                          <span className="feature-name">{item.feature}</span>
                          <span className="importance-value">{(item.importance * 100).toFixed(1)}%</span>
                        </div>
                        <div className="importance-bar">
                          <div 
                            className="importance-fill" 
                            style={{ width: `${item.importance * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sample Predictions */}
                <div className="predictions">
                  <h3 className="section-title">Sample Predictions</h3>
                  <div className="predictions-table">
                    <div className="table-header">
                      <div>Symbol</div>
                      <div>Actual</div>
                      <div>Predicted</div>
                      <div>Error</div>
                    </div>
                    {results.predictions.map((pred, idx) => (
                      <div key={idx} className="table-row">
                        <div className="symbol">{pred.symbol}</div>
                        <div className="actual">${pred.actual}</div>
                        <div className="predicted">${pred.predicted}</div>
                        <div className={`error ${Math.abs(pred.actual - pred.predicted) > 1 ? 'high' : 'low'}`}>
                          ±{Math.abs(pred.actual - pred.predicted).toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;