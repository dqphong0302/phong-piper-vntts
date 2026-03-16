"""
Valtec ONNX Engine — Integrates into Piper TTSEngine
Handles Valtec ONNX models alongside Piper ONNX models.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

# Add valtec-tts to path for text processing
sys.path.insert(0, "/root/opt/valtec-tts")

logger = logging.getLogger(__name__)

# Valtec model config path
VALTEC_CONFIG = Path("/root/.cache/valtec_tts/models/vits-vietnamese/config.json")


class ValtecOnnxVoice:
    """Single Valtec speaker loaded from ONNX."""

    def __init__(self, onnx_path: str, speaker: str, sample_rate: int = 24000):
        self.speaker = speaker
        self.sample_rate = sample_rate

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 4
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path, sess_opts, providers=["CPUExecutionProvider"]
        )
        logger.info("Loaded Valtec ONNX: %s (%s)", speaker, onnx_path)

        # Lazy-load text processor
        self._text_ready = False
        self._config = None

    def _ensure_text_pipeline(self):
        """Lazy-init text processing (imports are slow)."""
        if self._text_ready:
            return
        if VALTEC_CONFIG.exists():
            with open(VALTEC_CONFIG, 'r') as f:
                self._config = json.load(f)
        else:
            self._config = {"data": {"add_blank": True}}
        self._text_ready = True

    def synthesize(self, text: str, speed: float = 1.0) -> tuple:
        """Synthesize text to audio. Returns (audio_array, sample_rate)."""
        self._ensure_text_pipeline()

        from src.vietnamese.text_processor import process_vietnamese_text
        from src.vietnamese.phonemizer import text_to_phonemes, VIPHONEME_AVAILABLE
        from src.text import cleaned_text_to_sequence
        from src.nn import commons

        add_blank = self._config['data'].get('add_blank', True)

        normalized = process_vietnamese_text(text)
        phones, tones, word2ph = text_to_phonemes(normalized, use_viphoneme=VIPHONEME_AVAILABLE)
        phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phones, tones, "VI")

        if add_blank:
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            lang_ids = commons.intersperse(lang_ids, 0)

        seq_len = len(phone_ids)
        length_scale = 1.0 / speed if speed > 0 else 1.0

        outputs = self.session.run(
            ["audio"],
            {
                "x": np.array([phone_ids], dtype=np.int64),
                "x_lengths": np.array([seq_len], dtype=np.int64),
                "tone": np.array([tone_ids], dtype=np.int64),
                "language": np.array([lang_ids], dtype=np.int64),
                "bert": np.zeros((1, 1024, seq_len), dtype=np.float32),
                "ja_bert": np.zeros((1, 768, seq_len), dtype=np.float32),
                "onnx::Cast_6": np.array(0.667, dtype=np.float64),   # noise_scale
                "onnx::Cast_7": np.array(length_scale, dtype=np.float64),
                "onnx::Cast_8": np.array(0.8, dtype=np.float64),     # noise_scale_w
                "onnx::Cast_9": np.array(0.0, dtype=np.float64),     # sdp_ratio
            }
        )

        audio = outputs[0][0, 0]
        return audio, self.sample_rate


def is_valtec_model(config: dict) -> bool:
    """Check if a model config belongs to Valtec."""
    return config.get("model") == "valtec-tts"
