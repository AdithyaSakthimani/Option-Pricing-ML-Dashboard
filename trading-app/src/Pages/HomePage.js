import React, { useState, useEffect, useContext } from 'react';
import '../Styles/HomePage.css'; 
import NoteContext from '../Context/NoteContext';
import heroImg from '../images/hero_img_2.jpg'; 
import growthImg from '../images/feature_img_1.jpg';
import growthImg2 from '../images/feature_img_2.jpg';
import growthImg3 from '../images/feature_img_4.jpg';
import growthImg4 from '../images/feature_img_5.jpg'; 
import {Infinity,Rocket,Users,CircleCheckBig} from 'lucide-react'
import { Link } from 'react-router-dom';
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
  <Link to="/options" className="btn btn-primary">AI Simulation</Link>
  <Link to="/pastanalysis" className="btn btn-secondary"> Past Analysis </Link>
</div>
        </div>
        {/* Hero Image Placeholder */}
      </section>

      {/* Features Section */}
      <section className="features" id="features">
        <div className="features-container">
          <h2 className="my-section-title">Powerful Trading Features</h2>
          
          <div className="features-grid">
            <div className="feature-card">
                <img src ={growthImg} className='feature_img_1'/>
              <h3 className='my-grid-head'>Real-Time Analytics</h3>
              <p>Advanced market analysis with real-time options data, volatility tracking, and trend predictions powered by AI algorithms.</p>
            </div>
            
            <div className="feature-card">
              <img src ={growthImg2} className='feature_img_1'/>
              <h3 className='my-grid-head'>Advanced Charting</h3>
              <p>Professional-grade charts with technical indicators, options flow visualization, and customizable trading strategies.</p>
            </div>
            
            
            <div className="feature-card">
              <img src ={growthImg3} className='feature_img_1'/>
              <h3 className='my-grid-head'>Lightning Fast Execution</h3>
              <p>Ultra-low latency trading with direct market access and smart order routing for optimal execution.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
  <div className="particles">
    <div className="particle"></div>
    <div className="particle green"></div>
    <div className="particle red"></div>
    <div className="particle pulse"></div>
    <div className="particle"></div>
    <div className="particle green pulse"></div>
    <div className="particle red"></div>
    <div className="particle"></div>
  </div>

  <div className="stats-container">
    <h2 className="section-title-stats">Ready to Scale. Built to Deliver.</h2>

    <div className="stats-grid">
      <div className="stat-item">
        <span className="stat-number"><CircleCheckBig size ={55}/></span>
        <span className="stat-label">Infrastructure Ready</span>
      </div>

      <div className="stat-item">
        <span className="stat-number"><Users size={55} /></span>
        <span className="stat-label">First Users Coming Soon</span>
      </div>

      <div className="stat-item">
        <span className="stat-inf"><Infinity size = {55}/></span>
        <span className="stat-label">Growth Potential</span>
      </div>

      <div className="stat-item">
        <span className="stat-number"><Rocket size = {55}/></span>
        <span className="stat-label">We're Just Getting Started</span>
      </div>
    </div>
  </div>
</section>



      {/* Additional Features */}
      <section className="features">
  <div className="features-container">
    <h2 className="my-section-title">Powered by Proven ML Models</h2>

    <div className="features-grid-ai">
      <div className="feature-card">
        <div className="feature-icon purple">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="feature-icon-svg">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v6m0 6v6" />
            <path d="m21 12-6 0m-6 0-6 0" />
            <path d="m21 3-3 3-3-3" />
            <path d="m21 21-3-3-3 3" />
            <path d="m3 3 3 3-3 3" />
            <path d="m3 21 3-3 3 3" />
          </svg>
        </div>
        <h3><span className="text-purple">XGBoost</span> Predictor</h3>
        <p>Extreme Gradient Boosting algorithm optimized for speed and performance to uncover complex patterns and forecast market behavior.</p>
      </div>

      <div className="feature-card">
        <div className="feature-icon green">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="feature-icon-svg">
            <path d="M9 12l2 2 4-4" />
            <path d="M21 12c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1" />
            <path d="M3 12c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1" />
            <path d="M12 21c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1" />
            <path d="M12 3c.552 0 1-.448 1-1s-.448-1-1-1-1 .448-1 1 .448 1 1 1" />
          </svg>
        </div>
        <h3><span className="text-green">Linear Regression</span> Engine</h3>
        <p>Simple yet powerful statistical model that captures linear relationships in financial trends to assist in pricing and forecasting.</p>
      </div>

      <div className="feature-card">
        <div className="feature-icon blue">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="feature-icon-svg">
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="6" />
            <circle cx="12" cy="12" r="2" />
          </svg>
        </div>
        <h3><span className="text-blue">Random Forest</span> Classifier</h3>
        <p>Ensemble model that leverages multiple decision trees to reduce variance and enhance predictive accuracy across volatile markets.</p>
      </div>

      <div className="feature-card">
        <div className="feature-icon pink">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="feature-icon-svg">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <h3><span className="text-pink">Gradient Boosting</span> Engine</h3>
        <p>Boosted decision tree models that optimize prediction by correcting errors of prior iterations—ideal for nuanced market movements.</p>
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
        <li><Link to="/">Home</Link></li>
        <li><Link to="/options">AI Simulation</Link></li>
        <li><Link to="/train">Model Trainer</Link></li>
        <li><Link to="/pastanalysis">Past Analysis</Link></li>
      </ul>
    </div>

    <div className="footer-section">
      <h4>Account</h4>
      <ul>
        <li><Link to="/signin">Sign In</Link></li>
        <li><Link to="/signup">Sign Up</Link></li>
      </ul>
    </div>

    <div className="footer-section">
      <h4>Resources</h4>
      <ul>
        <li><a href="https://www.investopedia.com/options-basics-tutorial-4583012" target="_blank" rel="noopener noreferrer">Options Tutorial</a></li>
        <li><a href="https://www.cboe.com/" target="_blank" rel="noopener noreferrer">CBOE Education</a></li>
        <li><a href="https://www.optionstrading.org/" target="_blank" rel="noopener noreferrer">OptionsTrading.org</a></li>
      </ul>
    </div>

    <div className="footer-section">
      <h4>Support</h4>
      <ul>
        <li><a href="mailto:support@tradepro.ai">Contact Us</a></li>
        <li><a href="#status">System Status</a></li>
        <li><a href="#feedback">Send Feedback</a></li>
      </ul>
    </div>
  </div>

  <div className="footer-bottom">
    <p>&copy; 2025 DerivIQ. All rights reserved. Options trading involves risk and is not suitable for all investors.</p>
  </div>
</footer>

    </div>
  );
};

export default HomePage;