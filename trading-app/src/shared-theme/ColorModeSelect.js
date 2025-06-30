import * as React from 'react';
import { useColorScheme } from '@mui/material/styles';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import NoteContext from '../Context/NoteContext';
import { useContext, useEffect } from 'react';

export default function ColorModeSelect(props) {
  const { darkMode, setDarkMode } = useContext(NoteContext);
  const { mode, setMode } = useColorScheme();

  // 1️⃣ Sync MUI mode when darkMode changes
  useEffect(() => {
    if (darkMode === true && mode !== 'dark') {
      setMode('dark');
    } else if (darkMode === false && mode !== 'light') {
      setMode('light');
    }
  }, [darkMode]);

  // 2️⃣ Optionally, sync darkMode when MUI mode changes
  useEffect(() => {
    if (mode === 'dark' && !darkMode) {
      setDarkMode(true);
    } else if (mode === 'light' && darkMode) {
      setDarkMode(false);
    }
  }, [mode]);

  if (!mode) {
    return null;
  }

  return (
    <Select
      value={mode}
      onChange={(event) => setMode(event.target.value)}
      SelectDisplayProps={{
        'data-screenshot': 'toggle-mode',
      }}
      {...props}
    >
      <MenuItem value="system">System</MenuItem>
      <MenuItem value="light">Light</MenuItem>
      <MenuItem value="dark">Dark</MenuItem>
    </Select>
  );
}
