import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import NoteState from './Context/NoteState';
import Navbar from './Components/Navbar';
import {
  BrowserRouter as Router,
  Route,
  Routes,
} from 'react-router-dom';
import SignUp from './Sign_Up/SignUp';
import SignInSide from './Signup_Side/SignInSide';
import SignUpSide from './Signup_Side/SignUpSide';
import HomePage from './Pages/HomePage';
import MainArea from './Pages/MainArea';
import PastAnalysis from './Pages/PastAnalysis';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <NoteState>
      <Router>
      <Navbar/>
      <Routes>
          <Route path="/options" element={<MainArea/>} />
          <Route path="/" element={<HomePage/>} />
          <Route path="/train" element={<App/>} />
          <Route path="/signup" element={<SignUpSide/>} />
          <Route path="/signin" element = {<SignInSide/>}/>
          <Route path="/pastanalysis" element = {<PastAnalysis/>}/>
        </Routes>
      </Router>
    </NoteState>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
