# MCP Voice Assistant Setup Guide

This guide will help you set up and run the MCP Voice Assistant that enables voice interaction with Claude.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A microphone and speakers for audio input/output

## Installation

1. Create a new directory for your project and navigate to it:
   ```bash
   mkdir mcp-voice-assistant
   cd mcp-voice-assistant
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install mcp sounddevice numpy SpeechRecognition gtts playsound
   ```

   Note: Some additional dependencies may be required depending on your system:
   - For SpeechRecognition to work, you may need to install PyAudio
   - Some systems require additional portaudio development libraries

## Running the Voice Assistant

1. Save the code from the "MCP Voice Mode Server" artifact as `voice_server.py`

2. Run the server:
   ```bash
   python voice_server.py
   ```

3. Install the MCP server in Claude Desktop:
   ```bash
   mcp install voice_server.py --name "Voice Assistant"
   ```

4. Alternatively, test the server using the MCP Inspector:
   ```bash
   mcp dev voice_server.py
   ```

## Using Voice Mode

1. Once the server is running and connected to Claude, type "enter voice mode" in your conversation.

2. Claude will activate the voice prompt, greeting you and explaining how to use voice mode.

3. The system will:
   - Listen for your speech
   - Transcribe it to text for Claude
   - Convert Claude's response to speech

4. To exit voice mode, simply say "exit voice mode" during your conversation.

## Troubleshooting

- **Microphone access issues**: Ensure your application has permission to access your microphone.
- **Speech recognition errors**: Check your internet connection, as Google's speech recognition service requires internet access.
- **Audio output problems**: Verify your system's sound settings and ensure the default output device is selected.

## Extending the Voice Assistant

This is a basic prototype that you can extend in various ways:
- Add different TTS engines for better quality speech
- Implement speech recognition streaming for real-time transcription
- Add voice customization options
- Implement wake word detection
