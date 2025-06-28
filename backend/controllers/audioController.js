const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const AudioFile = require("../models/AudioFile");

const processAudio = async (req, res) => {
    try {
        const filePath = req.file.path;
        const originalName = req.file.originalname;

        // Processed File Output
        const processedPath = `uploads/processed_${Date.now()}.mp3`;

        // Save Metadata
        const newAudio = new AudioFile({
            originalName,
            filePath,
            processedPath,
        });

        // Process Audio with FFmpeg
        ffmpeg(filePath)
            .audioBitrate(128)
            .save(processedPath)
            .on("end", async () => {
                newAudio.processedPath = processedPath;

                // Retrieve metadata (duration, format, bitrate)
                ffmpeg.ffprobe(processedPath, async (err, metadata) => {
                    if (!err) {
                        const format = metadata.format;
                        newAudio.metadata = {
                            duration: format.duration,
                            format: format.format_name,
                            bitrate: format.bit_rate / 1000,
                        };

                        await newAudio.save(); // Save to MongoDB
                        fs.unlinkSync(filePath); // Remove the uploaded file
                        res.status(200).json({ message: "Audio processed successfully", data: newAudio });
                    } else {
                        throw new Error("Failed to retrieve metadata");
                    }
                });
            })
            .on("error", (err) => {
                res.status(500).json({ error: "Failed to process audio", details: err.message });
            });
    } catch (error) {
        res.status(500).json({ error: "An error occurred", details: error.message });
    }
};

const getAudioDetails = async (req, res) => {
    try {
        const { id } = req.params;
        const audioFile = await AudioFile.findById(id);

        if (!audioFile) {
            return res.status(404).json({ error: "Audio file not found" });
        }

        res.status(200).json(audioFile);
    } catch (error) {
        res.status(500).json({ error: "An error occurred", details: error.message });
    }
};

module.exports = { processAudio, getAudioDetails };
