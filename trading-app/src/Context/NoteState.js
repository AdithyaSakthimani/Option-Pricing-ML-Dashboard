import React,{useState , useEffect} from 'react';
import NoteContext from './NoteContext';


function NoteState(props) {
  const [darkMode, setDarkMode] = useState(() => {
    const stored = localStorage.getItem('darkMode');
    return stored !== null ? JSON.parse(stored) : false;
  });
  const [userId , setUserId] = useState(()=>{
    const stored = localStorage.getItem('userId') ; 
    return stored !== null ? JSON.parse(stored) : null ; 
  })
  const [activeItem, setActiveItem] = useState(() => {
    const stored = localStorage.getItem('activeItem');
    return stored !== null ? JSON.parse(stored) : 'home';
  });
  const [username, setUsername] = useState(() => {
    const stored = localStorage.getItem('username');
    return stored !== null ? JSON.parse(stored) : '';
  });
  const[email, setEmail] = useState(() => {
    const stored = localStorage.getItem('email');
    return stored !== null ? JSON.parse(stored) : '';}) 
  const[isAuthenticated, setIsAuthenticated] = useState(() => {
    const stored = localStorage.getItem('isAuthenticated');
    return stored !== null ? JSON.parse(stored) : false;
  });
  useEffect(() => {
    localStorage.setItem('activeItem', JSON.stringify(activeItem));
  }, [activeItem]);
    useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);
  useEffect(() => {
  localStorage.setItem('userId', JSON.stringify(userId));
}, [userId]);
  useEffect(() => {
    localStorage.setItem('username', JSON.stringify(username));
  }, [username]);

  useEffect(() => {
    localStorage.setItem('isAuthenticated',
      JSON.stringify(isAuthenticated));
  }, [isAuthenticated]);
  useEffect(() => {
    localStorage.setItem('email',
      JSON.stringify(email));
  }, [email]);
  const mockAnalysisData = {
  mse: 0.00234,
  r2: 0.9847,
  lastPrice: 152.75,
  impliedVolatility: 0.2847,
  greeks: {
    delta: 0.6543,
    gamma: 0.0234,
    theta: -0.0456,
    vega: 0.2134,
    rho: 0.1234
  },
  weightageFactors: [
    { factor: 'Implied Volatility', weight: 35.2, color: '#8b5cf6' },
    { factor: 'Time to Expiry', weight: 28.7, color: '#06b6d4' },
    { factor: 'Strike Price', weight: 18.9, color: '#10b981' },
    { factor: 'Interest Rate', weight: 12.1, color: '#f59e0b' },
    { factor: 'Dividend Yield', weight: 5.1, color: '#ef4444' }
  ],
  residualData: [
    { x: 1, residual: 0.002 }, { x: 2, residual: -0.001 }, { x: 3, residual: 0.003 },
    { x: 4, residual: -0.002 }, { x: 5, residual: 0.001 }, { x: 6, residual: -0.003 },
    { x: 7, residual: 0.002 }, { x: 8, residual: 0.001 }, { x: 9, residual: -0.001 },
    { x: 10, residual: 0.002 }, { x: 11, residual: -0.002 }, { x: 12, residual: 0.001 }
  ]
};
  const companies = [
    { symbol: 'AAPL', name: 'Apple Inc.' },
    { symbol: 'TSLA', name: 'Tesla Inc.' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.' },
    { symbol: 'MSFT', name: 'Microsoft Corp.' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.' },
    { symbol: 'META', name: 'Meta Platforms Inc.' },
    { symbol: 'NFLX', name: 'Netflix Inc.' },
    { symbol: 'AMD', name: 'Advanced Micro Devices' },
    { symbol: 'CRM', name: 'Salesforce Inc.' }
  ];
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

  // Model options
  const models = [
  { id: 'linear-regression', name: 'Linear Regression' },
  { id: 'random-forest', name: 'Random Forest Regressor' },
  { id: 'gradient-boosting', name: 'Gradient Boosting Regressor' },
  { id: 'xgboost', name: 'XGBoost Regressor' }
  ];
  const [analyses, setAnalyses] = useState([]);
  const option_types=[
  { id: 'call', name: 'Call Option' },
  { id: 'put', name: 'Put Option' }
  ]
   const [showResults, setShowResults] = useState(false);
   const [loading, setLoading] = useState(false);
   const [formData, setFormData] = useState({
     companyName: '',
     modelName: '',
     expiryDate: ''
   });
   const[model , setModel] = useState('') ; 
   const [data,setData] = useState(mockAnalysisData);
  return(
    <NoteContext.Provider  value={{ darkMode,setDarkMode ,username, setUsername,isAuthenticated, setIsAuthenticated,showResults,setShowResults,loading,setLoading,formData,setFormData , data , setData , companies , models , option_types , activeItem, setActiveItem,analyses, setAnalyses,userId , setUserId,setEmail , email , model , setModel }}>
        {props.children}
      </NoteContext.Provider>
  )
}

export default NoteState
