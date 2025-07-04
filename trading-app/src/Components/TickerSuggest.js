import React, { useState, useEffect, useRef } from 'react';
import { Building2, Search, CheckCircle, XCircle, TrendingUp, AlertTriangle } from 'lucide-react';
import '../Styles/TickerInput.css'; // Assuming you have a CSS file for styles

const TickerInput = ({ 
  value, 
  onChange, 
  isQuickMode, 
  placeholder = "e.g., AAPL, MSFT, TSLA",
  onValidationChange 
}) => {
  const [suggestions, setSuggestions] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isValid, setIsValid] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationMessage, setValidationMessage] = useState('');
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  
  const inputRef = useRef(null);
  const suggestionsRef = useRef(null);
  const debounceRef = useRef(null);

  // Extended list of popular tickers with company names and sectors
  const popularTickers = [
    { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', sector: 'Technology' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'TSLA', name: 'Tesla Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'META', name: 'Meta Platforms Inc.', sector: 'Technology' },
    { symbol: 'NVDA', name: 'NVIDIA Corporation', sector: 'Technology' },
    { symbol: 'NFLX', name: 'Netflix Inc.', sector: 'Communication Services' },
    { symbol: 'BABA', name: 'Alibaba Group Holding Ltd.', sector: 'Consumer Discretionary' },
    { symbol: 'V', name: 'Visa Inc.', sector: 'Financial Services' },
    { symbol: 'JPM', name: 'JPMorgan Chase & Co.', sector: 'Financial Services' },
    { symbol: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare' },
    { symbol: 'WMT', name: 'Walmart Inc.', sector: 'Consumer Staples' },
    { symbol: 'PG', name: 'Procter & Gamble Co.', sector: 'Consumer Staples' },
    { symbol: 'UNH', name: 'UnitedHealth Group Inc.', sector: 'Healthcare' },
    { symbol: 'HD', name: 'Home Depot Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'DIS', name: 'Walt Disney Co.', sector: 'Communication Services' },
    { symbol: 'MA', name: 'Mastercard Inc.', sector: 'Financial Services' },
    { symbol: 'PYPL', name: 'PayPal Holdings Inc.', sector: 'Financial Services' },
    { symbol: 'ADBE', name: 'Adobe Inc.', sector: 'Technology' },
    { symbol: 'CRM', name: 'Salesforce Inc.', sector: 'Technology' },
    { symbol: 'INTC', name: 'Intel Corporation', sector: 'Technology' },
    { symbol: 'AMD', name: 'Advanced Micro Devices Inc.', sector: 'Technology' },
    { symbol: 'COIN', name: 'Coinbase Global Inc.', sector: 'Financial Services' },
    { symbol: 'UBER', name: 'Uber Technologies Inc.', sector: 'Technology' },
    { symbol: 'LYFT', name: 'Lyft Inc.', sector: 'Technology' },
    { symbol: 'SNAP', name: 'Snap Inc.', sector: 'Communication Services' },
    { symbol: 'TWTR', name: 'Twitter Inc.', sector: 'Communication Services' },
    { symbol: 'ZOOM', name: 'Zoom Video Communications Inc.', sector: 'Technology' },
    { symbol: 'ROKU', name: 'Roku Inc.', sector: 'Communication Services' },
    { symbol: 'SQ', name: 'Block Inc.', sector: 'Technology' },
    { symbol: 'SHOP', name: 'Shopify Inc.', sector: 'Technology' },
    { symbol: 'SPOT', name: 'Spotify Technology S.A.', sector: 'Communication Services' },
    { symbol: 'PINS', name: 'Pinterest Inc.', sector: 'Communication Services' },
    { symbol: 'DOCU', name: 'DocuSign Inc.', sector: 'Technology' },
    { symbol: 'ZM', name: 'Zoom Video Communications Inc.', sector: 'Technology' },
    { symbol: 'PLTR', name: 'Palantir Technologies Inc.', sector: 'Technology' },
    { symbol: 'RBLX', name: 'Roblox Corporation', sector: 'Communication Services' },
    { symbol: 'RIVN', name: 'Rivian Automotive Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'LCID', name: 'Lucid Group Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'F', name: 'Ford Motor Company', sector: 'Consumer Discretionary' },
    { symbol: 'GM', name: 'General Motors Company', sector: 'Consumer Discretionary' },
    { symbol: 'NIO', name: 'NIO Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'XPEV', name: 'XPeng Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'LI', name: 'Li Auto Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'ABNB', name: 'Airbnb Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'DASH', name: 'DoorDash Inc.', sector: 'Consumer Discretionary' },
    { symbol: 'SNOW', name: 'Snowflake Inc.', sector: 'Technology' },
    { symbol: 'DDOG', name: 'Datadog Inc.', sector: 'Technology' },
    { symbol: 'CRWD', name: 'CrowdStrike Holdings Inc.', sector: 'Technology' },
    { symbol: 'ZS', name: 'Zscaler Inc.', sector: 'Technology' },
    { symbol: 'OKTA', name: 'Okta Inc.', sector: 'Technology' }
  ];

  // Simulate ticker validation (in real app, this would be an API call)
  const validateTicker = async (ticker) => {
    if (!ticker || ticker.length < 1) {
      setIsValid(null);
      setValidationMessage('');
      return;
    }

    setIsValidating(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const isValidTicker = popularTickers.some(t => t.symbol === ticker.toUpperCase());
    
    setIsValid(isValidTicker);
    setValidationMessage(
      isValidTicker 
        ? 'Valid ticker symbol' 
        : 'Invalid ticker symbol. Please check and try again.'
    );
    setIsValidating(false);
    
    // Notify parent component about validation status
    if (onValidationChange) {
      onValidationChange(isValidTicker);
    }
  };

  // Debounced validation
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    
    debounceRef.current = setTimeout(() => {
      validateTicker(value);
    }, 800);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [value]);

  // Filter suggestions based on input
  const filterSuggestions = (input) => {
    if (!input || input.length < 1) return [];
    
    const filtered = popularTickers.filter(ticker => 
      ticker.symbol.toLowerCase().includes(input.toLowerCase()) ||
      ticker.name.toLowerCase().includes(input.toLowerCase())
    ).slice(0, 8); // Limit to 8 suggestions
    
    return filtered;
  };

  const handleInputChange = (e) => {
    const newValue = e.target.value.toUpperCase();
    onChange(newValue);
    
    const filtered = filterSuggestions(newValue);
    setSuggestions(filtered);
    setIsOpen(filtered.length > 0 && newValue.length > 0);
    setHighlightedIndex(-1);
  };

  const handleSuggestionClick = (ticker) => {
    onChange(ticker.symbol);
    setIsOpen(false);
    setSuggestions([]);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (!isOpen) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightedIndex(prev => prev > 0 ? prev - 1 : prev);
        break;
      case 'Enter':
        e.preventDefault();
        if (highlightedIndex >= 0) {
          handleSuggestionClick(suggestions[highlightedIndex]);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setHighlightedIndex(-1);
        break;
    }
  };

  const handleBlur = () => {
    // Delay hiding suggestions to allow for click events
    setTimeout(() => {
      setIsOpen(false);
      setHighlightedIndex(-1);
    }, 200);
  };

  const handleFocus = () => {
    if (value && suggestions.length > 0) {
      setIsOpen(true);
    }
  };

  const getValidationIcon = () => {
    if (isValidating) {
      return <div className="validation-spinner" />;
    }
    if (isValid === true) {
      return <CheckCircle className="validation-icon valid" size={16} />;
    }
    if (isValid === false) {
      return <XCircle className="validation-icon invalid" size={16} />;
    }
    return null;
  };

  return (
    <div className="ticker-input-container">
      <label className="form-label">
        <Building2 size={18} />
        {isQuickMode ? 'Enter Ticker Symbol' : 'Stock Symbol'}
      </label>
      
      <div className="ticker-input-wrapper">
        <div className="input-with-icon">
          <Search className="search-icon" size={16} />
          <input
            ref={inputRef}
            type="text"
            className={`ticker-input ${isValid === false ? 'invalid' : ''} ${isValid === true ? 'valid' : ''}`}
            value={value}
            placeholder={placeholder}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            onFocus={handleFocus}
            autoComplete="off"
            spellCheck="false"
          />
          <div className="validation-indicator">
            {getValidationIcon()}
          </div>
        </div>
        
        {isOpen && suggestions.length > 0 && (
          <div className="suggestions-dropdown" ref={suggestionsRef}>
            <div className="suggestions-header">
              <TrendingUp size={14} />
              <span>Popular Tickers</span>
            </div>
            {suggestions.map((ticker, index) => (
              <div
                key={ticker.symbol}
                className={`suggestion-item ${index === highlightedIndex ? 'highlighted' : ''}`}
                onClick={() => handleSuggestionClick(ticker)}
              >
                <div className="suggestion-symbol">{ticker.symbol}</div>
                <div className="suggestion-details">
                  <div className="suggestion-name">{ticker.name}</div>
                  <div className="suggestion-sector">{ticker.sector}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {validationMessage && (
        <div className={`validation-message ${isValid ? 'valid' : 'invalid'}`}>
          {isValid === false && <AlertTriangle size={14} />}
          {isValid === true && <CheckCircle size={14} />}
          <span>{validationMessage}</span>
        </div>
      )}
    </div>
  );
};

export default TickerInput;