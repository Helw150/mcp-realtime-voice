from mcp.server.fastmcp import FastMCP, Context
import sys
import pyaudio
import numpy as np
import wave
import tempfile
import os
import speech_recognition as sr
from gtts import gTTS
import platform
import subprocess
import json
import functools
import warnings
from typing import List, Dict, Optional, Union, Tuple, NamedTuple

# Create an MCP server with required dependencies
mcp = FastMCP("Voice Assistant", dependencies=[
    "pyaudio", 
    "numpy", 
    "wave", 
    "speech_recognition", 
    "gtts", 
    "onnxruntime"
])

# Global variables for VAD
VAD_ENABLED = True
DEFAULT_VAD_THRESHOLD = 0.2
SILENCE_DURATION = 3  # seconds of silence before stopping recording

# ----- VAD Implementation -----

class VadOptions(NamedTuple):
    """VAD options."""
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    window_size_samples: int = 1536
    speech_pad_ms: int = 400


class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def get_initial_state(self, batch_size: int):
        h = np.zeros((2, batch_size, 64), dtype=np.float32)
        c = np.zeros((2, batch_size, 64), dtype=np.float32)
        return h, c

    def __call__(self, x, state, sr: int):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        if len(x.shape) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {len(x.shape)}")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        h, c = state

        ort_inputs = {
            "input": x,
            "h": h,
            "c": c,
            "sr": np.array(sr, dtype="int64"),
        }

        out, h, c = self.session.run(None, ort_inputs)
        state = (h, c)

        return out, state


class VADIterator:
    def __init__(self, model, sampling_rate=16000, threshold=0.5):
        self.model = model
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.reset_states()

    def reset_states(self):
        self.state = self.model.get_initial_state(batch_size=1)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        if not isinstance(x, np.ndarray):
            try:
                x = np.frombuffer(x, dtype=np.float32)
            except:
                x = np.frombuffer(x, dtype=np.int16).astype(np.float32) / 32768.0

        out, self.state = self.model(x, self.state, self.sampling_rate)
        speech_prob = out[0][0]
        
        # Simple triggering logic
        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            return {'start': self.current_sample if not return_seconds else self.current_sample / self.sampling_rate}
        
        if speech_prob < self.threshold and self.triggered:
            self.triggered = False
            return {'end': self.current_sample if not return_seconds else self.current_sample / self.sampling_rate}
            
        self.current_sample += len(x)
        return {}


# Function to download the Silero VAD ONNX model
def download_vad_model(model_path):
    """Download Silero VAD model if not already present."""
    if os.path.exists(model_path):
        return True
        
    import urllib.request
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        print(f"Downloading Silero VAD model to {model_path}...")
        urllib.request.urlretrieve(
            "https://huggingface.co/spaces/freddyaboulton/mini-omni2-webrtc/resolve/main/utils/assets/silero_vad.onnx?download=true",
            model_path
        )
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


# Singleton to load the VAD model once
@functools.lru_cache(maxsize=1)
def load_silero_vad():
    """Load the Silero VAD model. Returns None if model can't be loaded."""
    try:
        # Define model path in user directory to avoid permission issues
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "silero_vad")
        model_path = os.path.join(model_dir, "silero_vad.onnx")
        
        # Download model if not present
        if not download_vad_model(model_path):
            return None
            
        # Create the model
        return SileroVADModel(model_path)
    except Exception as e:
        print(f"Error loading Silero VAD model: {e}")
        return None


# ----- Tools for speech interaction -----

@mcp.tool()
async def list_audio_devices(ctx: Context) -> str:
    """
    Lists all available audio devices and their properties.
    Useful for debugging microphone access issues.
    """
    ctx.info("Listing available audio devices...")
    
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
        ctx.info(f"Found {device_count} audio devices")
        for dev in device_info:
            if dev["is_input"]:
                ctx.info(f"Input device {dev['index']}: {dev['name']} (channels: {dev['maxInputChannels']})")
        
        # Terminate PyAudio
        p.terminate()
        
        # Return formatted list
        return json.dumps(device_info, indent=2)
        
    except Exception as e:
        ctx.error(f"Error listing audio devices: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def listen_for_speech(ctx: Context) -> str:
    """
    Records audio from the microphone and transcribes it to text.
    Returns the transcribed text or an error message.
    
    With VAD enabled, it will automatically detect when speech starts and ends
    instead of using a fixed recording duration.
    """
    global VAD_ENABLED
    ctx.info("Listening for speech...")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1536  # Optimal size for VAD (32ms at 16kHz)
        
        # Load VAD model if enabled
        vad_model = None
        vad_iterator = None
        
        if VAD_ENABLED:
            vad_model = load_silero_vad()
            if vad_model:
                vad_iterator = VADIterator(vad_model, sampling_rate=RATE)

        # Inform user we're starting to record
        if not VAD_ENABLED:
            RECORD_SECONDS = 5
            ctx.info(f"Recording started. Please speak now (recording for {RECORD_SECONDS} seconds)...")
        else:
            ctx.info("Recording started. Please speak now (will auto-detect end of speech)...")
        
        # Open stream with error handling
        try:
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK)
        except Exception as e:
            ctx.error(f"Error opening audio stream: {str(e)}")
            p.terminate()
            return f"Error opening audio stream: {str(e)}"
        
        # Record audio
        frames = []
        
        if not VAD_ENABLED:
            # Traditional fixed-time recording
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        else:
            # VAD-based recording
            speech_started = False
            silence_counter = 0
            max_silence_chunks = int(SILENCE_DURATION * RATE / CHUNK)
            pre_buffer = []  # To capture audio just before speech detection
            pre_buffer_size = int(0.5 * RATE / CHUNK)  # 0.5 seconds of pre-buffer
            
            # Initial listening phase to detect speech
            ctx.info("Waiting for speech to begin...")
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Keep a rolling buffer of recent audio
                pre_buffer.append(data)
                if len(pre_buffer) > pre_buffer_size:
                    pre_buffer.pop(0)
                    
                # Process with VAD - convert audio data to float array
                audio_numpy = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                # Get speech detection from VAD
                speech_dict = vad_iterator(audio_numpy, return_seconds=False)
                    
                if speech_dict and speech_dict.get('start', False):
                    speech_started = True
                    ctx.info("Speech detected, recording...")
                    # Add pre-buffer to capture the beginning of speech
                    frames.extend(pre_buffer)
                    frames.append(data)
                    break
            
            # Continue recording until silence is detected
            if speech_started:
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                        
                    # Process with VAD
                    audio_numpy = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                    # Get the prediction from the model
                    out, _ = vad_model(audio_numpy, vad_model.get_initial_state(1), RATE)
                    speech_prob = out[0][0]
                        
                    if speech_prob < DEFAULT_VAD_THRESHOLD:
                        silence_counter += 1
                    else:
                        silence_counter = 0
                        
                    # If sufficient silence, stop recording
                    if silence_counter > max_silence_chunks:
                        ctx.info("End of speech detected.")
                        print("End of speech detected.", file=sys.stderr)
                        break
                                    
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Reset VAD states if used
        if vad_iterator:
            vad_iterator.reset_states()
        
        # Terminate PyAudio
        p.terminate()
        
        # Check if we got any frames
        if not frames:
            ctx.error("No audio data captured")
            return "Error: No audio data captured"
        
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
                ctx.info(f"Transcribed text: {transcript}")
            except sr.UnknownValueError:
                ctx.error("Speech recognition could not understand audio")
                transcript = "I couldn't understand what you said. Can you please try again?"
            except sr.RequestError as e:
                ctx.error(f"Could not request results from Google Speech Recognition service: {e}")
                transcript = "Sorry, there was an error with the speech recognition service."
            
        # Clean up temp file
        os.unlink(temp_filename)
        
        return transcript
        
    except Exception as e:
        ctx.error(f"Error during speech recognition: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def speak_text(text: str, ctx: Context) -> str:
    """
    Converts text to speech and plays it.
    
    Args:
        text: The text to convert to speech
        
    Returns:
        Status message
    """
    ctx.info(f"Speaking: {text}")
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            temp_filename = temp_audio.name
        
        # Convert text to speech
        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)
        
        # Create the lock file path in /tmp directory
        lock_file = "/tmp/speech_lock"
        
        # Play the audio based on platform with bash lock
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            # Using flock for locking and chaining with &&
            cmd = f"flock {lock_file} -c 'afplay {temp_filename} --rate 2' && rm -f {temp_filename}"
            subprocess.Popen(cmd, shell=True)
        elif system == 'Linux':
            # Using flock on Linux
            cmd = f"flock {lock_file} -c 'mpg123 {temp_filename}' && rm -f {temp_filename}"
            subprocess.Popen(cmd, shell=True)
        elif system == 'Windows':
            # Windows doesn't easily support flock, so we'll fall back to the original method
            from winsound import PlaySound, SND_FILENAME
            PlaySound(temp_filename, SND_FILENAME)
            os.unlink(temp_filename)
        else:
            ctx.error(f"Unsupported platform: {system}")
            return f"Error: Unsupported platform {system}"
        
        return "Successfully played speech"
        
    except Exception as e:
        ctx.error(f"Error during text-to-speech: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def voice_mode(ctx: Context) -> str:
    """
    Enter a voice conversation mode where the assistant listens for speech
    and responds with speech.
    """
    return """
    You are now in voice conversation mode. I'll handle the speech input and output for you.
    
    In this mode, you should:
    1. Use the listen_for_speech tool to get user input
    2. Process the input and generate your response
    3. Use the speak_text tool to convert your response to speech. Split your speech into chunks of 20 words or less to avoid latency.
    4. Repeat steps 1-3 until the user says "exit voice mode" or similar
    
    Begin by greeting the user and explaining that they're in voice conversation mode.
    Tell them they can say "exit voice mode" to return to text input.
    
    With Voice Activity Detection (VAD) enabled, the system will automatically detect when the user 
    starts and stops speaking, making the interaction more natural.
    """


# Run the server
if __name__ == "__main__":
    # Try to initialize the VAD model
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        
        vad_model = load_silero_vad()
        if vad_model:
            print("Silero VAD (ONNX model) loaded successfully")
            # Test the model to ensure it works
            test_audio = np.zeros(1536, dtype=np.float32)
            state = vad_model.get_initial_state(1)
            out, _ = vad_model(test_audio, state, 16000)
            print("Model test passed!")
        else:
            print("Failed to load Silero VAD model. Falling back to fixed-time recording.")
            VAD_ENABLED = False
    except ImportError as e:
        print(f"ONNX Runtime not installed. Error: {e}")
        print("Falling back to fixed-time recording")
        VAD_ENABLED = False
    except Exception as e:
        print(f"Error initializing Silero VAD: {e}")
        print("Falling back to fixed-time recording")
        VAD_ENABLED = False
    
    mcp.run()
