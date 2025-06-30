import React, { useState, useEffect, useContext } from 'react';
import { TrendingUp, Calculator, BarChart3, Moon, Sun, Activity, Target, Zap, Search, Calendar, Building2, Brain, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, BarChart, Bar } from 'recharts';
import '../Styles/MainArea.css'; // Import your CSS styles
import ParaInput from '../Components/ParaInput';
import NoteContext from '../Context/NoteContext';
import {GreekCard,MetricCard } from '../Components/MainComponents';
import DisplayPage from '../Components/DisplayPage';
export default function MainFile() {
  const { darkMode, setDarkMode, showResults } = useContext(NoteContext);
  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>

      {!showResults ? (
        /* Input Form */
        <ParaInput/>
      ) : (
        /* Results Section */
        <DisplayPage/>
      )}
    </div>
  );
}