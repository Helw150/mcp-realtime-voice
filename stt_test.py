import asyncio
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import speech_recognition as sr

async def listen_for_speech():
    """
    Records audio from the microphone and transcribes it to text.
    Returns the transcribed text or an error message.
    """
    print("Listening for speech...")
    
    # Record audio
    fs = 44100  # Sample rate
    seconds = 5  # Recording duration
    
    try:
        # Inform user we're starting to record
        print("Recording started. Please speak now...")
        
        # Record audio
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        # Save recording to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_filename = temp_audio.name
            
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(fs)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        
        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
            
        # Clean up temp file
        os.unlink(temp_filename)
        
        print(f"Transcribed: {transcript}")
        return transcript
        
    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        return f"Error: {str(e)}"

async def main():
    print("=== Speech-to-Text Test ===")
    print("This test will record 5 seconds of audio and attempt to transcribe it.")
    print("Press Enter to start the recording...")
    
    # Wait for user to press Enter
    input()
    
    # Try to transcribe speech
    transcript = await listen_for_speech()
    
    print("\nTest completed!")
    print(f"Final transcript: {transcript}")

if __name__ == "__main__":
    asyncio.run(main())
