import asyncio
import pyaudio
import numpy as np
import wave
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
import platform
import subprocess

async def list_audio_devices():
    """
    Lists all available audio devices and their properties.
    Useful for debugging microphone access issues.
    """
    print("Listing available audio devices...")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Get all audio devices
        device_count = p.get_device_count()
        
        # Format device info for better readability
        device_info = []
        for i in range(device_count):
            dev = p.get_device_info_by_index(i)
            device_info.append({
                "index": i,
                "name": dev["name"],
                "hostApi": dev["hostApi"],
                "maxInputChannels": dev["maxInputChannels"],
                "maxOutputChannels": dev["maxOutputChannels"],
                "defaultSampleRate": dev["defaultSampleRate"],
                "is_input": dev["maxInputChannels"] > 0,
                "is_output": dev["maxOutputChannels"] > 0
            })
        
        # Log detailed device information
        print(f"Found {device_count} audio devices")
        for dev in device_info:
            if dev["is_input"]:
                print(f"Input device {dev['index']}: {dev['name']} (channels: {dev['maxInputChannels']})")
        
        # Terminate PyAudio
        p.terminate()
        
        return device_info
        
    except Exception as e:
        print(f"Error listing audio devices: {str(e)}")
        return f"Error: {str(e)}"

async def listen_for_speech():
    """
    Records audio from the microphone and transcribes it to text.
    Returns the transcribed text or an error message.
    """
    print("Listening for speech...")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        
        # Inform user we're starting to record
        print("Recording started. Please speak now...")
        
        # Open stream
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        # Record audio
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Terminate PyAudio
        p.terminate()
        
        # Save recording to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_filename = temp_audio.name
            
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            
        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio_data = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_data)
                print(f"Transcribed text: {transcript}")
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                transcript = "I couldn't understand what you said. Can you please try again?"
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service: {e}")
                transcript = "Sorry, there was an error with the speech recognition service."
            
        # Clean up temp file
        os.unlink(temp_filename)
        
        return transcript
        
    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        return f"Error: {str(e)}"

async def speak_text(text):
    """
    Converts text to speech and plays it.
    
    Args:
        text: The text to convert to speech
        
    Returns:
        Status message
    """
    print(f"Speaking: {text}")
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_filename = temp_audio.name
        
        # Convert text to speech
        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)
        
        # Play the audio based on platform
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            subprocess.call(['afplay', temp_filename])
        elif system == 'Linux':
            subprocess.call(['mpg123', temp_filename])
        elif system == 'Windows':
            from winsound import PlaySound, SND_FILENAME
            PlaySound(temp_filename, SND_FILENAME)
        else:
            print(f"Unsupported platform: {system}")
            return f"Error: Unsupported platform {system}"
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        return "Successfully played speech"
        
    except Exception as e:
        print(f"Error during text-to-speech: {str(e)}")
        return f"Error: {str(e)}"

async def main():
    print("=== Simple Voice Assistant Test ===")
    print("This will test the listen and speak functionality in a loop.")
    print("Say 'exit' or 'quit' to end the test.")
    
    # Optionally list audio devices at startup
    devices = await list_audio_devices()
    print(f"Available input devices: {[d['index'] for d in devices if d['is_input']]}")
    
    await speak_text("Voice assistant test started. Please speak after the prompt.")
    
    while True:
        try:
            # Listen for user input
            user_input = await listen_for_speech()
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "stop", "end"]:
                await speak_text("Ending voice assistant test. Goodbye!")
                break
                
            # Simple response based on input
            response = f"You said: {user_input}. Please speak again or say exit to quit."
            await speak_text(response)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            await speak_text("I encountered an error. Let's try again.")

if __name__ == "__main__":
    asyncio.run(main())
