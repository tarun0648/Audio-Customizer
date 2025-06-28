import React, { useState } from "react";

const AudioEditor = () => {
    const [startTime, setStartTime] = useState(0);
    const [endTime, setEndTime] = useState(0);
    const [volume, setVolume] = useState(1);

    const handleTrim = () => {
        // TODO: Send trim request to backend
        console.log(`Trimming audio from ${startTime} to ${endTime}`);
    };

    const handleVolumeChange = (e) => {
        setVolume(e.target.value);
        // TODO: Update audio volume dynamically
    };

    return (
        <div className="audio-editor">
            <div>
                <label>Start Time:</label>
                <input type="number" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
            </div>
            <div>
                <label>End Time:</label>
                <input type="number" value={endTime} onChange={(e) => setEndTime(e.target.value)} />
            </div>
            <button onClick={handleTrim}>Trim Audio</button>

            <div>
                <label>Volume:</label>
                <input type="range" min="0" max="2" step="0.1" value={volume} onChange={handleVolumeChange} />
            </div>
        </div>
    );
};

export default AudioEditor;
