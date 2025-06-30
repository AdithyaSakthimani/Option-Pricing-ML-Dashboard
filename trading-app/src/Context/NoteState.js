import React,{useState , useEffect} from 'react';
import NoteContext from './NoteContext';


function NoteState(props) {
  const [darkMode, setDarkMode] = useState(() => {
    const stored = localStorage.getItem('darkMode');
    return stored !== null ? JSON.parse(stored) : false;
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
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

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

  // Model options
  const models = [
    { id: 'black-scholes', name: 'Black-Scholes Model' },
    { id: 'binomial', name: 'Binomial Model' },
    { id: 'monte-carlo', name: 'Monte Carlo Simulation' },
    { id: 'lstm', name: 'LSTM Neural Network' }
  ];
   const [showResults, setShowResults] = useState(false);
   const [loading, setLoading] = useState(false);
   const [formData, setFormData] = useState({
     companyName: '',
     modelName: '',
     expiryDate: ''
   });
   const [data] = useState(mockAnalysisData);
  return(
    <NoteContext.Provider  value={{ darkMode,setDarkMode ,username, setUsername,isAuthenticated, setIsAuthenticated,showResults,setShowResults,loading,setLoading,formData,setFormData , data , companies , models}}>
        {props.children}
      </NoteContext.Provider>
  )
}

export default NoteState
