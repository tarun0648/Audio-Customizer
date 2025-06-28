import React from "react";
import AudioPlayer from "react-h5-audio-player";
import "react-h5-audio-player/lib/styles.css";

const AudioPlayerComponent = ({ audioUrl }) => {
    return (
        <div className="audio-player">
            <AudioPlayer src={audioUrl} controls />
        </div>
    );
};

export default AudioPlayerComponent;
