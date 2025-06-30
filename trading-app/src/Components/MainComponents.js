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
export const GreekCard = ({  name, value, description }) => (
    <div className="greek-card">
      <div className="greek-name">{name}</div>
      <div className="greek-value">{value}</div>
      <div className="greek-description">{description}</div>
    </div>
  );


