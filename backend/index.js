import mongoose from 'mongoose';
import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import Prediction_Data_Schema from './MongoSchema.js';
import User_Schema from './UserSchema.js' ; 
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
dotenv.config();
const MONGO_URI = process.env.MONGO_URI;
const JWT_SECRET = process.env.JWT_SECRET || 'secret123';
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => {
  console.log('Connected to MongoDB');
})
.catch((error) => {
  console.error('Error connecting to MongoDB:', error);
});
const app = express();
app.use(cors());
app.use(express.json());
const PORT = process.env.PORT || 5000;
app.post('/sendData', async (req, res) => {
  try{const data = req.body;
  const predictionData = new Prediction_Data_Schema({
    stock: data.stock,
    model: data.model,
    formData: {
      bidPrice: data.bidPrice,
      askPrice: data.askPrice,
      strikePrice: data.strikePrice,
      expiryDate: data.expiryDate,
      impliedVolatility: data.impliedVolatility,
      riskFreeRate: data.riskFreeRate,
      timeToMaturity: data.timeToMaturity
    },
    prediction: {
      callPrice: data.callPrice,
      putPrice: data.putPrice,
      delta: data.delta,
      gamma: data.gamma,
      theta: data.theta,
      vega: data.vega
    },
    userId: data.userId 
  });
  await predictionData.save();
  res.status(200).send('Saved the data successfully');}
  catch (error) {
    console.error('Error saving data:', error);
    res.status(500).send('Error saving data ');

  }
});
app.post('/getData', async (req, res) => {
  try{const data = req.body ;
  const userName = data.user ; 
  const prediction = await Prediction_Data_Schema.find({userId:userName})
  res.status(200).json({
    message:'data fetched successfully',
    prediction})}
  catch(error){
    res.status(500).send('error fetching data',error)
  }
});
app.post('/signup', async (req, res) => {
  try {
    console.log('Signup request received:', req.body);
    const { name, email, password } = req.body;

    const existingUser = await User_Schema.findOne({ email });
    if (existingUser) return res.status(400).json({ message: 'Email already in use' });

    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = new User_Schema({
      name,
      email,
      password: hashedPassword
    });

    await newUser.save();
    res.status(201).json({ message: 'User registered successfully' });

  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ message: 'Signup failed', error: error.message });
  }
});
app.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User_Schema.findOne({ email });
    if (!user) return res.status(400).json({ message: 'Invalid email or password' });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: 'Invalid email or password' });

    const token = jwt.sign({ id: user._id }, JWT_SECRET, { expiresIn: '7d' });

    res.status(200).json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email
      }
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Login failed', error: error.message });
  }
});
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
