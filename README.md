# MCP Realtime Voice

Turn Claude into a voice assistant that listens and speaks. This MCP server handles speech recognition and text-to-speech for a natural voice interface.

> **⚠️ IMPORTANT:** 
> - This has only been tested on macOS.
> - Launch Claude from the terminal with the command below instead of clicking the app icon. This fixes microphone permission issues.

## Features

- Speech recognition with silence detection
- Text-to-speech for AI responses
- Voice Activity Detection using Silero
- Works on Windows, macOS, and Linux
- Audio device management
- Simple voice conversation interface

## Prerequisites

- Python 3.8+
- A microphone and speakers
- MCP client (Claude)

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/mcp-realtime-voice.git
   cd mcp-realtime-voice
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   System dependencies:
   - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
   - macOS: `brew install portaudio`

## Usage

### Starting the Server

```bash
python voice_server.py
```

### Connecting to Claude

1. Launch Claude from the terminal:
   ```bash
   # On macOS
   /Applications/Claude.app/Contents/MacOS/Claude
   
   # On Windows
   # Use the path to your Claude executable
   start "" "C:\Path\to\Claude.exe"
   ```

2. Install the MCP server:
   ```bash
   mcp install voice_server.py --name "Realtime Voice"
   ```

   Or test with the MCP Inspector:
   ```bash
   mcp dev voice_server.py
   ```

### Available Tools

- **list_audio_devices**: Shows all audio input/output devices
- **listen_for_speech**: Records and transcribes speech
- **speak_text**: Converts text to spoken audio
- **voice_mode**: Starts interactive voice conversation

### Voice Conversation Mode

To start:
1. Connect the MCP server to Claude
2. Ask Claude to "enter voice mode"
3. Start talking - the system will:
   - Listen for your speech
   - Detect when you finish speaking
   - Send transcribed text to Claude
   - Speak Claude's response

To exit, just say "exit voice mode" or "stop voice mode".

## Configuration

Edit these values in `voice_server.py` if needed:

- VAD_THRESHOLD: Voice detection sensitivity (default: 0.2)
- SILENCE_DURATION: Seconds of silence before recording stops (default: 3)
- Audio sample rate and format settings
