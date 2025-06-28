const mongoose = require('mongoose');

const AudioFileSchema = new mongoose.Schema({
  originalFilePath: {
    type: String,
    required: true,
  },
  processedFiles: [
    {
      filePath: String,
      description: String,
    },
  ],
  uploadedAt: {
    type: Date,
    default: Date.now,
  },
});

const AudioFile = mongoose.model('AudioFile', AudioFileSchema);
module.exports = AudioFile;
