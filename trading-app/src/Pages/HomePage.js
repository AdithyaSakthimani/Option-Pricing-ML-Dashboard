import React, { useState, useEffect, useContext } from 'react';
import '../Styles/HomePage.css'; 
import NoteContext from '../Context/NoteContext';
import heroImg from '../images/hero_img_2.jpg'; 
import growthImg from '../images/feature_img_1.jpg';
import growthImg2 from '../images/feature_img_2.jpg';
import growthImg3 from '../images/feature_img_4.jpg';
const HomePage = () => {
  const {darkMode, setDarkMode} = useContext(NoteContext);
  useEffect(() => {
    // Apply theme to document
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
    }
  }, [darkMode]);
  const [data, setData] = useState([]);
  // Simulate real-time data updates every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const now = new Date();
        const newData = {
          time: now.toLocaleTimeString().slice(0, 8),
          price: (100 + Math.random() * 10).toFixed(2),
        };
        const updated = [...prev, newData];
        return updated.length > 20 ? updated.slice(updated.length - 20) : updated;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);
  return (
    <div>

      {/* Hero Section */}
      <section className="hero">
        <img src={heroImg} className='hero-image'/>
        <div className='hero-main'><h2 className='hero-title'><span className='hero-color'>Master Options </span> Trading Like Never Before</h2>
        <p className='hero-para'>Advanced analytics, real-time data, and intelligent insights to maximize your options trading potential. Join thousands of successful traders using our platform.</p>
        
        <div className="cta-buttons">
          <a href="#" className="btn btn-primary">Start Trading Free</a>
          <a href="#" className="btn btn-secondary">Watch Demo</a>
        </div>
        </div>
        {/* Hero Image Placeholder */}
      </section>

      {/* Features Section */}
      <section className="features" id="features">
        <div className="features-container">
          <h2 className="section-title">Powerful Trading Features</h2>
          
          <div className="features-grid">
            <div className="feature-card">
                <img src ={growthImg} className='feature_img_1'/>
              <h3>Real-Time Analytics</h3>
              <p>Advanced market analysis with real-time options data, volatility tracking, and trend predictions powered by AI algorithms.</p>
            </div>
            
            <div className="feature-card">
              <img src ={growthImg2} className='feature_img_1'/>
              <h3>Advanced Charting</h3>
              <p>Professional-grade charts with technical indicators, options flow visualization, and customizable trading strategies.</p>
            </div>
            
            
            <div className="feature-card">
              <img src ={growthImg3} className='feature_img_1'/>
              <h3>Lightning Fast Execution</h3>
              <p>Ultra-low latency trading with direct market access and smart order routing for optimal execution.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
            <div class="particles">
        <div class="particle"></div>
        <div class="particle green"></div>
        <div class="particle red"></div>
        <div class="particle pulse"></div>
        <div class="particle"></div>
        <div class="particle green pulse"></div>
        <div class="particle red"></div>
        <div class="particle"></div>
    </div>
        <div className="stats-container">
          <h2 className="section-title-stats">Trusted by Traders Worldwide</h2>
          
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-number">50K+</span>
              <span className="stat-label">Active Traders</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-number">$2.5B</span>
              <span className="stat-label">Options Volume</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-number">99.9%</span>
              <span className="stat-label">Uptime</span>
            </div>
            
            <div className="stat-item">
              <span className="stat-number">24/7</span>
              <span className="stat-label">Market Coverage</span>
            </div>

          </div>
        </div>
      </section>


      {/* Additional Features */}
      <section className="features">
        <div className="features-container">
          <h2 className="section-title">Powered by Advanced AI Models</h2>
          
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon purple">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className='feature-icon-svg'>
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M12 1v6m0 6v6"/>
                  <path d="m21 12-6 0m-6 0-6 0"/>
                  <path d="m21 3-3 3-3-3"/>
                  <path d="m21 21-3-3-3 3"/>
                  <path d="m3 3 3 3-3 3"/>
                  <path d="m3 21 3-3 3 3"/>
                </svg>
              </div>
              <h3><span className="text-purple">Neural Network</span> Prediction</h3>
              <p>Deep learning models trained on millions of market data points to predict price movements and identify profitable options strategies.</p>
            </div>
            
            <div className="feature-card">
              <div className="feature-icon green">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className='feature-icon-svg'>
                  <path d="M9 12l2 2 4-4"/>
                  <path d="M21 12c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1"/>
                  <path d="M3 12c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1"/>
                  <path d="M12 21c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1"/>
                  <path d="M12 3c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1"/>
                </svg>
              </div>
              <h3><span className="text-green">Sentiment Analysis</span> Engine</h3>
              <p>Natural Language Processing models that analyze news, social media, and market sentiment to gauge market mood and predict volatility.</p>
            </div>
            
        
            
            <div className="feature-card">
              <div className="feature-icon purple">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className='feature-icon-svg'>
                  <circle cx="12" cy="12" r="10"/>
                  <circle cx="12" cy="12" r="6"/>
                  <circle cx="12" cy="12" r="2"/>
                </svg>
              </div>
              <h3><span className="text-purple">Adaptive Learning</span> System</h3>
              <p>Self-improving AI models that learn from your trading patterns and market changes to provide increasingly personalized recommendations.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-section">
            <h4>Platform</h4>
            <ul>
              <li><a href="#trading">Trading Platform</a></li>
              <li><a href="#mobile">Mobile Apps</a></li>
              <li><a href="#api">API Access</a></li>
              <li><a href="#tools">Trading Tools</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Markets</h4>
            <ul>
              <li><a href="#stocks">Stock Options</a></li>
              <li><a href="#etf">ETF Options</a></li>
              <li><a href="#index">Index Options</a></li>
              <li><a href="#futures">Futures Options</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Education</h4>
            <ul>
              <li><a href="#learn">Learning Center</a></li>
              <li><a href="#webinars">Webinars</a></li>
              <li><a href="#research">Market Research</a></li>
              <li><a href="#blog">Trading Blog</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Support</h4>
            <ul>
              <li><a href="#help">Help Center</a></li>
              <li><a href="#contact">Contact Us</a></li>
              <li><a href="#status">System Status</a></li>
              <li><a href="#feedback">Feedback</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; 2025 TradePro. All rights reserved. Trading involves risk and may not be suitable for all investors.</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;