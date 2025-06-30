import * as React from 'react';
import { useState } from 'react';
import {
  Box,
  Button,
  Card as MuiCard,
  Checkbox,
  FormLabel,
  FormControl,
  FormControlLabel,
  TextField,
  Typography,
  Alert,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './SignInCard.css';

const Card = styled(MuiCard)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignSelf: 'center',
  width: '100%',
  padding: theme.spacing(4),
  gap: theme.spacing(2),
  boxShadow:
    'hsla(220, 30%, 5%, 0.05) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.05) 0px 15px 35px -5px',
  [theme.breakpoints.up('sm')]: {
    width: '450px',
  },
  ...theme.applyStyles?.('dark', {
    boxShadow:
      'hsla(220, 30%, 5%, 0.5) 0px 5px 15px 0px, hsla(220, 25%, 10%, 0.08) 0px 15px 35px -5px',
  }),
}));

export default function SignUpCard() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
  });

  const [status, setStatus] = useState({ type: '', message: '' });

  const [errors, setErrors] = useState({
    name: '',
    email: '',
    password: '',
  });

  const validateInputs = () => {
    const { name, email, password } = formData;
    const newErrors = { name: '', email: '', password: '' };
    let isValid = true;

    if (!name.trim()) {
      newErrors.name = 'Name is required';
      isValid = false;
    }

    if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = 'Valid email required';
      isValid = false;
    }

    if (password.length < 6) {
      newErrors.password = 'Minimum 6 characters required';
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const handleChange = (e) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateInputs()) return;

    try {
      const res = await axios.post('http://localhost:5000/signup', formData);
      setStatus({ type: 'success', message: res.data.message });
    } catch (err) {
      setStatus({
        type: 'error',
        message: err.response?.data?.message || 'Signup failed',
      });
    }
  };

  return (
    <div className="main-container">
      <Card variant="outlined">
        <Typography component="h1" variant="h4">
          Sign Up
        </Typography>

        {status.message && (
          <Alert severity={status.type}>{status.message}</Alert>
        )}

        <Box
          component="form"
          onSubmit={handleSubmit}
          noValidate
          sx={{ display: 'flex', flexDirection: 'column', width: '100%', gap: 2 }}
        >
          <FormControl>
            <FormLabel htmlFor="name">Name</FormLabel>
            <TextField
              id="name"
              name="name"
              required
              fullWidth
              value={formData.name}
              onChange={handleChange}
              error={Boolean(errors.name)}
              helperText={errors.name}
            />
          </FormControl>

          <FormControl>
            <FormLabel htmlFor="email">Email</FormLabel>
            <TextField
              id="email"
              name="email"
              type="email"
              required
              fullWidth
              value={formData.email}
              onChange={handleChange}
              error={Boolean(errors.email)}
              helperText={errors.email}
            />
          </FormControl>

          <FormControl>
            <FormLabel htmlFor="password">Password</FormLabel>
            <TextField
              id="password"
              name="password"
              type="password"
              required
              fullWidth
              value={formData.password}
              onChange={handleChange}
              error={Boolean(errors.password)}
              helperText={errors.password}
            />
          </FormControl>

          <FormControlLabel
            control={<Checkbox value="remember" color="primary" />}
            label="Remember me"
          />

          <Button type="submit" fullWidth variant="contained">
            Create Account
          </Button>

          <Typography sx={{ textAlign: 'center' }}>
            Already have an account?{' '}
            <Link to="/signin">
              Log in
            </Link>
          </Typography>
        </Box>
      </Card>
    </div>
  );
}
