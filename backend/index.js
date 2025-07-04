import mongoose from 'mongoose';
import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import Prediction from './MongoSchema.js';
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


app.post('/save-analysis', async (req, res) => {
  try {
    const { userId, results } = req.body;

    if (!userId || !results) {
      return res.status(400).json({ message: 'Missing userId or results data' });
    }

    const newPrediction = new Prediction({
      userId,
      ...results 
    });

    await newPrediction.save();

    res.status(200).json({ message: 'Analysis saved successfully' });
  } catch (error) {
    console.error('Error saving analysis:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});
app.post('/get-user-analyses', async (req, res) => {
  try {
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({ message: 'Missing userId' });
    }

    const predictions = await Prediction.find({ userId }).sort({ createdAt: -1 });

    res.status(200).json({ message: 'Fetched predictions', data: predictions });
  } catch (error) {
    console.error('Error fetching analyses:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});
app.delete('/delete-analysis/:id', async (req, res) => {
  const { id } = req.params;
  const { userId } = req.body;

  if (!userId) {
    return res.status(400).json({ message: 'User ID required for deletion' });
  }

  try {
    const deleted = await Prediction.findOneAndDelete({ _id: id, userId });

    if (!deleted) {
      return res.status(404).json({ message: 'Analysis not found or not authorized' });
    }

    res.status(200).json({ message: 'Analysis deleted successfully' });
  } catch (error) {
    console.error('Error deleting analysis:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
