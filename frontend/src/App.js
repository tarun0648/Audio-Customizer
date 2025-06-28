import React from 'react';
import AudioUpload from './components/AudioUpload';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="header">
        <h1 className="title">Audio Customizer</h1>
      </header>
      <main className="container">
        <AudioUpload />
      </main>
      <footer className="footer">
        <p>Thank You For Visiting Our Site</p>
      </footer>
    </div>
  );
}

export default App;
