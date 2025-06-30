import React, { useState, useEffect, useContext } from 'react';
import { TrendingUp, Calculator, BarChart3, Moon, Sun, Activity, Target, Zap, Search, Calendar, Building2, Brain, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import NoteContext from '../Context/NoteContext';
import { GreekCard, MetricCard } from './MainComponents';
function DisplayPage() {
    const{darkMode, setDarkMode ,companies,models,showResults,setShowResults,loading,setLoading,formData,setFormData,data} = useContext(NoteContext) ; 
  const resetForm = () => {
    setShowResults(false);
    setFormData({
      companyName: '',
      modelName: '',
      expiryDate: ''
    });
  };
  const topFactor = data.weightageFactors.reduce((prev, current) => 
    (prev.weight > current.weight) ? prev : current
  );
  return (
    <>
              <div className="header-section">
                <div className="header-content">
                  <h1 className="main-title">Analysis Results</h1>
                  <p className="main-subtitle">
                    {formData.companyName} • {models.find(m => m.id === formData.modelName)?.name} • Expires: {new Date(formData.expiryDate).toLocaleDateString()}
                  </p>
                  <button className="back-button" onClick={resetForm}>
                    ← New Analysis
                  </button>
                </div>
              </div>
    
              <div className="main-content">
            
            {/* Key Metrics Grid */}
            <div className="metrics-grid">
              <MetricCard 
                title="Model Accuracy (R²)"
                value={`${(data.r2 * 100).toFixed(2)}%`}
                subtitle="Coefficient of Determination"
                icon={Target}
                trend={2.3}
              />
              <MetricCard 
                title="Mean Squared Error"
                value={data.mse.toFixed(6)}
                subtitle="Lower is better"
                icon={Activity}
                trend={-1.8}
              />
              <MetricCard
                title="Last Price"
                value={`$${data.lastPrice}`}
                subtitle="Current market price"
                icon={TrendingUp}
                trend={0.7}
              />
              <MetricCard 
                title="Implied Volatility"
                value={`${(data.impliedVolatility * 100).toFixed(2)}%`}
                subtitle="Market expectation"
                icon={Zap}
                trend={-0.5}
              />
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
                  value={data.greeks.delta.toFixed(4)}
                  description="Price sensitivity to underlying"
                />
                <GreekCard 
                  name="Gamma (Γ)" 
                  value={data.greeks.gamma.toFixed(4)}
                  description="Delta sensitivity to underlying"
                />
                <GreekCard 
                  name="Theta (Θ)" 
                  value={data.greeks.theta.toFixed(4)}
                  description="Time decay sensitivity"
                />
                <GreekCard 
                  name="Vega (ν)" 
                  value={data.greeks.vega.toFixed(4)}
                  description="Volatility sensitivity"
                />
                <GreekCard 
                  name="Rho (ρ)" 
                  value={data.greeks.rho.toFixed(4)}
                  description="Interest rate sensitivity"
                />
              </div>
            </div>
    
            {/* Weightage Factors */}
            <div className="section-card">
              <div className="section-header">
                <BarChart3 size={24} />
                <h2>Factor Weightage Analysis</h2>
                <div className="top-factor-badge">
                  Most Important: {topFactor.factor} ({topFactor.weight}%)
                </div>
              </div>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data.weightageFactors} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#333' : '#e0e0e0'} />
                    <XAxis 
                      dataKey="factor" 
                      stroke={darkMode ? '#9ca3af' : '#6b7280'}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      interval={0}
                    />
                    <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: darkMode ? '#1f2937' : '#ffffff',
                        border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
                        borderRadius: '8px',
                        color: darkMode ? '#ffffff' : '#000000'
                      }}
                    />
                    <Bar dataKey="weight" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
    
            {/* Residual Plot */}
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
    
                </div>
            </>
  )
}

export default DisplayPage
