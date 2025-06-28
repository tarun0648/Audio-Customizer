const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const mongoose = require('mongoose');
const { exec } = require('child_process');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const port = 5001;

// Middleware
app.use(cors());
app.use(express.json());
// Serve uploads folder as static files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));


// MongoDB Connection
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch((err) => console.error('MongoDB connection error:', err));

// MongoDB Schema
const audioSchema = new mongoose.Schema({
  originalFilePath: String,
  processedFiles: [
    {
      feature: String,
      filePath: String,
      createdAt: { type: Date, default: Date.now },
    },
  ],
});

const Audio = mongoose.model('Audio', audioSchema);

// Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
});
const upload = multer({ storage });

// Upload route
app.post('/upload', upload.single('audio'), async (req, res) => {
  try {
    const audio = new Audio({
      originalFilePath: req.file.path,
      processedFiles: [],
    });
    const savedAudio = await audio.save();
    res.json({ message: 'File uploaded successfully', filePath: req.file.path, audioId: savedAudio._id });
  } catch (err) {
    console.error('Error saving file metadata:', err);
    res.status(500).json({ error: 'Failed to save file metadata' });
  }
});

// Utility function to process audio
const processAudio = async (audioId, feature, command, res) => {
  try {
    const audio = await Audio.findById(audioId);
    if (!audio) return res.status(404).json({ error: 'Audio not found' });

    exec(command, async (err, stdout, stderr) => {
      if (err) {
        console.error(`Error processing ${feature}:`, err);
        return res.status(500).json({ error: `Failed to process ${feature}` });
      }
      const processedFilePath = command.split(' ').pop(); // Extract output file path
      audio.processedFiles.push({ feature, filePath: processedFilePath });
      await audio.save();
      res.json({ message: `${feature} applied successfully`, filePath: processedFilePath });
    });
  } catch (err) {
    console.error('Error processing audio:', err);
    res.status(500).json({ error: 'Failed to process audio' });
  }
};

// Feature Routes
app.post('/trim', (req, res) => {
  const { audioId, start, end } = req.body;
  const outputFile = `uploads/trimmed-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -ss ${start} -to ${end} -c copy ${outputFile}`;
  processAudio(audioId, 'trim', command, res);
});

app.post('/change-speed', (req, res) => {
  const { audioId, speed } = req.body;
  const outputFile = `uploads/speed-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -filter:a "atempo=${speed}" -vn ${outputFile}`;
  processAudio(audioId, 'change-speed', command, res);
});

app.post('/adjust-volume', (req, res) => {
  const { audioId, volume } = req.body;
  const outputFile = `uploads/volume-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -filter:a "volume=${volume}" ${outputFile}`;
  processAudio(audioId, 'adjust-volume', command, res);
});

app.post('/reverse', (req, res) => {
  const { audioId } = req.body;
  const outputFile = `uploads/reversed-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -filter_complex "areverse" ${outputFile}`;
  processAudio(audioId, 'reverse', command, res);
});

app.post('/apply-fade', (req, res) => {
  const { audioId, fadeInDuration, fadeOutDuration } = req.body;
  const outputFile = `uploads/faded-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -af "afade=t=in:st=0:d=${fadeInDuration},afade=t=out:st=0:d=${fadeOutDuration}" ${outputFile}`;
  processAudio(audioId, 'apply-fade', command, res);
});

app.post('/generate-waveform', (req, res) => {
  const { audioId } = req.body;
  const outputFile = `uploads/waveform-${Date.now()}.png`;
  const command = `ffmpeg -i ${req.body.filePath} -filter_complex "showwavespic=s=640x120" -frames:v 1 ${outputFile}`;
  processAudio(audioId, 'generate-waveform', command, res);
});

app.post('/increase-pitch', (req, res) => {
  const { audioId, pitch } = req.body;
  const outputFile = `uploads/pitch-${Date.now()}.mp3`;
  const command = `ffmpeg -i ${req.body.filePath} -filter:a "asetrate=44100*${pitch},atempo=1/${pitch}" ${outputFile}`;
  processAudio(audioId, 'increase-pitch', command, res);
});

app.listen(port, () => console.log(`Server running on port ${port}`));
