import asyncio
import tempfile
import os
from gtts import gTTS
import platform
import subprocess

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
    print("=== TTS Test ===")
    
    # Test phrases
    phrases = [
        "Hello world! This is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "This is the final test phrase. Thank you for testing the speech system."
    ]
    
    for phrase in phrases:
        print(f"\nTesting phrase: {phrase}")
        result = await speak_text(phrase)
        print(f"Result: {result}")
        # Wait a moment between phrases
        await asyncio.sleep(1)
    
    print("\nTTS test completed!")

if __name__ == "__main__":
    asyncio.run(main())
