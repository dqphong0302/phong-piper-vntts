"""
Vietnamese TTS Engine using Piper-TTS
Compatible with Windows, Linux, macOS
"""
import json
import logging
import wave
import io
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from valtec_onnx_engine import ValtecOnnxVoice, is_valtec_model

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine using Piper-TTS for Vietnamese."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.voices: Dict[str, PiperVoice] = {}
        self.valtec_voices = {}
        self._load_available_models()
        self._preload_all_models()

    def _load_available_models(self) -> None:
        """Scan models directory and load model configurations."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        for onnx_file in self.models_dir.glob("*.onnx"):
            model_name = onnx_file.stem
            config_file = onnx_file.with_suffix(".onnx.json")

            # Load config if exists
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            else:
                config = {"audio": {"sample_rate": 22050}}

            self.models[model_name] = {
                "path": str(onnx_file),
                "config_path": str(config_file) if config_file.exists() else None,
                "config": config,
            }
            logger.info(f"Found model: {model_name}")

        # Load Valtec ONNX models separately
        for name, info in list(self.models.items()):
            if is_valtec_model(info["config"]):
                try:
                    sr = info["config"].get("sample_rate", 24000)
                    self.valtec_voices[name] = ValtecOnnxVoice(
                        info["path"], info["config"].get("speaker", name), sr
                    )
                    logger.info(f"Loaded Valtec ONNX model: {name}")
                except Exception as e:
                    logger.error(f"Failed to load Valtec model {name}: {e}")


    def _preload_all_models(self) -> None:
        """Pre-load all models into memory for instant inference."""
        for model_name in list(self.models.keys()):
            if model_name in self.valtec_voices:
                continue
            try:
                self.get_voice(model_name)
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
        logger.info(f"Pre-loaded {len(self.voices)}/{len(self.models)} voice models")

    def get_voice(self, model_name: str) -> PiperVoice:
        """Get or create PiperVoice instance (lazy loading)."""
        if model_name not in self.voices:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")

            model_info = self.models[model_name]
            model_path = model_info["path"]
            config_path = model_info.get("config_path")

            logger.info(f"Loading Piper model: {model_name}")

            self.voices[model_name] = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=False,
            )

        return self.voices[model_name]

    def list_voices(self) -> List[dict]:
        """List all available voice models."""
        voices = []
        for name, info in self.models.items():
            config = info["config"]
            voices.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "language": config.get("espeak", {}).get("voice", "vi"),
                "sample_rate": config.get("audio", {}).get("sample_rate", 22050),
            })
        return voices

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Input text in Vietnamese
            voice: Voice model name (uses first available if not specified)
            speed: Speech speed multiplier (1.0 = normal)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Select voice
        if voice is None:
            voice = next(iter(self.models.keys()))
        if voice not in self.models:
            raise ValueError(f"Voice not found: {voice}. Available: {list(self.models.keys())}")

        # Route Valtec ONNX models
        if voice in self.valtec_voices:
            audio, sr = self.valtec_voices[voice].synthesize(text, speed=speed)
            return audio, sr

        config = self.models[voice]["config"]
        sample_rate = config.get("audio", {}).get("sample_rate", 22050)

        try:
            piper_voice = self.get_voice(voice)

            # Create SynthesisConfig with speed control
            length_scale = 1.0 / speed if speed > 0 else 1.0
            syn_config = SynthesisConfig(length_scale=length_scale)

            # Generate audio (Piper returns already-normalized float32 in [-1,1])
            all_audio = []
            for chunk in piper_voice.synthesize(text, syn_config=syn_config):
                if hasattr(chunk, 'audio_float_array'):
                    all_audio.append(chunk.audio_float_array)
                elif hasattr(chunk, 'audio_int16_array'):
                    all_audio.append(chunk.audio_int16_array.astype(np.float32) / 32767.0)

            if not all_audio:
                return np.array([], dtype=np.float32), sample_rate

            audio = np.concatenate(all_audio)
            
            return audio, sample_rate

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise RuntimeError(f"Failed to generate speech: {e}")

    def audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to WAV bytes."""
        # Normalize and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
