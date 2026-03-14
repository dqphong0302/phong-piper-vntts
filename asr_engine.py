"""
ASR Engine — Speech-to-Text using Sherpa-ONNX Whisper
Supports file upload transcription with timestamps.
"""

import logging
import os
import wave
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


class ASREngine:
    """Whisper-based ASR engine using sherpa-onnx."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.recognizer = None
        self.model_name = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        whisper_dir = self.model_dir / "sherpa-onnx-whisper-medium"
        if not whisper_dir.exists():
            logger.warning(f"ASR model not found at {whisper_dir}")
            return

        encoder = str(whisper_dir / "medium-encoder.int8.onnx")
        decoder = str(whisper_dir / "medium-decoder.int8.onnx")
        tokens = str(whisper_dir / "medium-tokens.txt")

        if not all(os.path.exists(f) for f in [encoder, decoder, tokens]):
            logger.error("Missing ASR model files")
            return

        try:
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                num_threads=2,
                language="vi",
                task="transcribe",
            )
            self.model_name = "whisper-medium"
            logger.info(f"ASR model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.recognizer is not None

    def _convert_to_wav16k(self, audio_bytes: bytes, filename: str = "") -> str:
        """Convert any audio format to 16kHz mono WAV using ffmpeg."""
        suffix = Path(filename).suffix if filename else ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name

        out_path = in_path + ".wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path,
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                 out_path],
                capture_output=True, check=True, timeout=60
            )
            return out_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise RuntimeError(f"Cannot convert audio: {e}")
        finally:
            os.unlink(in_path)

    def _read_wav(self, wav_path: str) -> tuple:
        """Read WAV file and return (samples, sample_rate)."""
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1, "Must be mono"
            assert wf.getsampwidth() == 2, "Must be 16-bit"
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sample_rate

    def transcribe(self, audio_bytes: bytes, filename: str = "",
                   language: str = "vi") -> dict:
        """
        Transcribe audio bytes to text.
        Returns dict with 'text', 'language', 'duration'.
        """
        if not self.is_ready:
            raise RuntimeError("ASR engine not loaded")

        wav_path = self._convert_to_wav16k(audio_bytes, filename)
        try:
            samples, sample_rate = self._read_wav(wav_path)
            duration = len(samples) / sample_rate

            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()

            logger.info(f"ASR result ({duration:.1f}s): {text[:80]}...")

            return {
                "text": text,
                "language": language,
                "duration": round(duration, 2),
                "model": self.model_name,
            }
        finally:
            os.unlink(wav_path)

    def get_info(self) -> dict:
        """Get ASR engine info."""
        return {
            "ready": self.is_ready,
            "model": self.model_name,
            "engine": "sherpa-onnx-whisper",
            "supported_languages": ["vi", "en"],
        }
