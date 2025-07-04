import React, { useContext } from 'react'
import NoteContext from '../Context/NoteContext';
export const MetricCard = ({ title, value, subtitle, icon: Icon, trend }) => (
    <div className="metric-card">
      <div className="metric-header">
        <div className="metric-icon">
          <Icon size={20} />
        </div>
        <div className="metric-title">{title}</div>
      </div>
      <div className="metric-value">{value}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
      {trend && (
        <div className={`metric-trend ${trend > 0 ? 'positive' : 'negative'}`}>
          {trend > 0 ? '+' : ''}{trend}%
        </div>
      )}
    </div>
  );
// Helper function to format scientific notation
const formatScientificNotation = (value) => {
  if (!value) return '0.0000';
  
  // Convert string to number if it's a string
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  
  // Check if the number is very small or very large
  if (Math.abs(numValue) < 0.0001 || Math.abs(numValue) >= 10000) {
    // Format in scientific notation with 4 decimal places
    return numValue.toExponential(4);
  } else {
    // Format as regular decimal with 4 decimal places
    return numValue.toFixed(4);
  }
};

// Helper function to get color based on Greek type and value
const getGreekColor = (name, value) => {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  
  if (name.includes('Delta')) {
    return numValue > 0 ? '#10b981' : '#ef4444'; // Green for positive, red for negative
  } else if (name.includes('Gamma')) {
    return numValue > 0 ? '#3b82f6' : '#6b7280'; // Blue for positive, gray for zero/negative
  } else if (name.includes('Theta')) {
    return numValue < 0 ? '#f59e0b' : '#10b981'; // Amber for negative (time decay), green for positive
  } else if (name.includes('Vega')) {
    return numValue > 0 ? '#8b5cf6' : '#6b7280'; // Purple for positive, gray for zero/negative
  } else if (name.includes('Rho')) {
    return numValue > 0 ? '#06b6d4' : '#ef4444'; // Cyan for positive, red for negative
  }
  
  return '#6b7280'; // Default gray
};

export const GreekCard = ({ name, value, description }) => {
  const formattedValue = formatScientificNotation(value);
  const valueColor = getGreekColor(name, value);
  
  return (
    <div className="greek-card">
      <div className="greek-name">{name}</div>
      <div 
        className="greek-value" 
        style={{ color: valueColor }}
        title={`Raw value: ${value}`} // Show original value on hover
      >
        {formattedValue}
      </div>
      <div className="greek-description">{description}</div>
    </div>
  );
};

