import React, { useState } from 'react';
import axios from 'axios';

function AudioUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFilePath, setUploadedFilePath] = useState('');
  const [audioId, setAudioId] = useState('');
  const [outputFiles, setOutputFiles] = useState([]);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('audio', selectedFile);

    try {
      const response = await axios.post('http://localhost:5001/upload', formData);
      setUploadedFilePath(response.data.filePath);
      setAudioId(response.data.audioId);
      alert('File uploaded successfully!');
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload file');
    }
  };

  const processAudio = async (endpoint, data) => {
    try {
      const response = await axios.post(`http://localhost:5001/${endpoint}`, { ...data, audioId });
      setOutputFiles((prev) => [...prev, { feature: endpoint, filePath: response.data.filePath }]);
      alert(`${endpoint.replace('-', ' ')} applied successfully!`);
    } catch (error) {
      console.error(`Error applying ${endpoint}:`, error);
      alert(`Failed to apply ${endpoint}`);
    }
  };

  return (
    <div className="audio-uploader">
      <h1>Audio Customizer</h1>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Audio</button>

      {uploadedFilePath && (
        <div>
          <h3>Uploaded Audio: {uploadedFilePath}</h3>
          <button onClick={() => processAudio('trim', { filePath: uploadedFilePath, start: 30, end: 60 })}>
            Trim Audio (30s to 60s)
          </button>
          <button onClick={() => processAudio('change-speed', { filePath: uploadedFilePath, speed: 1.5 })}>
            Change Speed (1.5x)
          </button>
          <button onClick={() => processAudio('adjust-volume', { filePath: uploadedFilePath, volume: 1.5 })}>
            Adjust Volume (+50%)
          </button>
          <button onClick={() => processAudio('reverse', { filePath: uploadedFilePath })}>
            Reverse Audio
          </button>
          <button onClick={() => processAudio('apply-fade', { filePath: uploadedFilePath, fadeInDuration: 5, fadeOutDuration: 5 })}>
            Apply Fade (5s In/Out)
          </button>
          <button onClick={() => processAudio('generate-waveform', { filePath: uploadedFilePath })}>
            Generate Waveform
          </button>
          <button onClick={() => processAudio('increase-pitch', { filePath: uploadedFilePath, pitch: 1.2 })}>
            Increase Pitch (1.2x)
          </button>
        </div>
      )}

      {outputFiles.length > 0 && (
        <div>
          <h3>Processed Files</h3>
          <ul>
            {outputFiles.map((file, index) => (
              <li key={index}>
                {file.feature}: <a href={`http://localhost:5001/${file.filePath}`} download>Download</a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default AudioUpload;
