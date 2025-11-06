#!/usr/bin/env python3
"""
Audio service for MAESTRO.
Handles microphone input and Whisper transcription.
"""

import sounddevice as sd
import numpy as np
import whisper

def record_audio(duration=5, samplerate=16000):
    print(f"Recording {duration} seconds of audio...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(recording)

def transcribe_audio(model_name="base"):
    print("Loading Whisper model...")
    model = whisper.load_model(model_name)

    print("Recording audio...")
    audio_data = record_audio()

    print("Transcribing...")
    result = model.transcribe(audio_data)
    print("Transcription:", result["text"])
    return result["text"]

if __name__ == "__main__":
    transcribe_audio()
