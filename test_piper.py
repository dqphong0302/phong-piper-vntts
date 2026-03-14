"""Test script for Piper TTS"""
from piper import PiperVoice

# Load model
voice = PiperVoice.load(
    "models/calmwoman3688.onnx",
    config_path="models/calmwoman3688.onnx.json"
)

# Generate audio
print("Generating audio...")
chunks = list(voice.synthesize("xin chào"))

print(f"Number of chunks: {len(chunks)}")
if chunks:
    chunk = chunks[0]
    print(f"Chunk type: {type(chunk)}")
    print(f"Chunk dir: {[a for a in dir(chunk) if not a.startswith('_')]}")
    
    # Try to print attributes
    for attr in dir(chunk):
        if not attr.startswith('_'):
            try:
                val = getattr(chunk, attr)
                print(f"  {attr}: type={type(val).__name__}, len={len(val) if hasattr(val, '__len__') else 'N/A'}")
            except Exception as e:
                print(f"  {attr}: error={e}")
