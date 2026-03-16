"""
Vietnamese & English TTS API - FastAPI Application
Configuration via environment variables (see .env.example)
"""
import logging
import subprocess
import tempfile
import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from tts_engine import TTSEngine
from vietnamese_processor import process_vietnamese_text

# ── Configuration from environment ──
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "minhkhang")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese Speech API",
    description="Bidirectional Speech API: TTS (Piper ONNX) + ASR (Whisper) for Vietnamese",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS engine
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(Path(__file__).parent / "models")))
STATIC_DIR = Path(__file__).parent / "static"
tts_engine = TTSEngine(str(MODELS_DIR))

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def apply_pitch(wav_bytes: bytes, pitch: float, sample_rate: int = 22050) -> bytes:
    """Apply pitch shift using ffmpeg asetrate filter.
    pitch=1.0 means no change, >1.0 higher, <1.0 lower.
    Uses asetrate to change pitch then atempo to compensate speed."""
    if pitch == 1.0:
        return wav_bytes

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        f_in.write(wav_bytes)
        in_path = f_in.name
    out_path = in_path.replace(".wav", "_pitch.wav")

    try:
        new_rate = int(sample_rate * pitch)
        tempo = 1.0 / pitch
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-af", f"asetrate={new_rate},atempo={tempo:.4f},aresample={sample_rate}",
             out_path],
            capture_output=True, check=True, timeout=30
        )
        with open(out_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Pitch shift failed: {e}, returning original")
        return wav_bytes
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def convert_format(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convert WAV to mp3/m4a/ogg using ffmpeg. Returns (bytes, media_type)."""
    if fmt == "wav":
        return wav_bytes, "audio/wav"

    # Format: (codec_args, extension, media_type, sample_rate)
    codec_map = {
        "mp3": (["libmp3lame", "-b:a", "128k"], ".mp3", "audio/mpeg", 44100),
        "m4a": (["aac", "-b:a", "128k"], ".m4a", "audio/mp4", 44100),
        "aac": (["aac", "-b:a", "64k"], ".aac", "audio/aac", 44100),
        "opus": (["libopus", "-b:a", "64k"], ".ogg", "audio/ogg", 48000),
        "ogg": (["libopus", "-b:a", "64k"], ".ogg", "audio/ogg", 48000),
        "amr": (["libopencore_amrnb", "-b:a", "12.2k"], ".amr", "audio/amr", 8000),
    }
    if fmt not in codec_map:
        return wav_bytes, "audio/wav"

    codec_args, ext, media, target_sr = codec_map[fmt]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        in_path = f.name
    out_path = in_path.replace(".wav", ext)

    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", str(target_sr), "-ac", "1", "-codec:a"] + codec_args
        if fmt in ("ogg", "opus"):
            cmd += ["-f", "ogg"]
        if fmt == "aac":
            cmd += ["-f", "adts"]
        cmd += [out_path]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        with open(out_path, "rb") as f:
            return f.read(), media
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


class TTSRequest(BaseModel):
    """Request body for TTS synthesis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice model ID")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Pitch shift (0.5=low, 1.0=normal, 2.0=high)")
    noise_scale: Optional[float] = Field(None, ge=0.0, le=1.0, description="Noise scale for variation")
    noise_w: Optional[float] = Field(None, ge=0.0, le=1.0, description="Noise width")


class VoiceInfo(BaseModel):
    """Voice information."""
    id: str
    name: str
    language: str
    sample_rate: int


@app.get("/", tags=["GUI"], include_in_schema=False)
async def root():
    """Redirect to GUI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Vietnamese TTS API",
        "version": "2.0.0",
        "voices_loaded": len(tts_engine.models),
        "features": ["pitch", "speed", "noise_scale", "multi_format", "vi_text_processing", "asr"],
    }


@app.get("/voices", response_model=list[VoiceInfo], tags=["Voices"])
async def list_voices():
    """List all available voice models."""
    return tts_engine.list_voices()


@app.post("/tts", tags=["TTS"])
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text.
    Returns WAV audio file. Supports pitch shifting via ffmpeg.
    """
    try:
        # Preprocess Vietnamese text
        processed_text = process_vietnamese_text(request.text)
        logger.info(f"Text preprocessing: {request.text[:50]}... -> {processed_text[:50]}...")

        audio, sample_rate = tts_engine.synthesize(
            text=processed_text,
            voice=request.voice,
            speed=request.speed,
            noise_scale=request.noise_scale,
            noise_w=request.noise_w,
        )

        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="No audio generated")

        wav_bytes = tts_engine.audio_to_wav_bytes(audio, sample_rate)

        # Apply pitch shift if needed
        if request.pitch != 1.0:
            wav_bytes = apply_pitch(wav_bytes, request.pitch, sample_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts", tags=["TTS"])
async def synthesize_speech_get(
    text: str = Query(..., min_length=1, max_length=5000, description="Text to synthesize"),
    voice: Optional[str] = Query(None, description="Voice model ID"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speech speed"),
    pitch: float = Query(1.0, ge=0.5, le=2.0, description="Pitch shift"),
):
    """Synthesize speech from text (GET method)."""
    request = TTSRequest(text=text, voice=voice, speed=speed, pitch=pitch)
    return await synthesize_speech(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True, log_level=LOG_LEVEL)


# === OpenAI-compatible TTS endpoint ===

class OpenAISpeechRequest(BaseModel):
    """OpenAI-compatible speech request."""
    model: str = Field(default="tts-1", description="TTS model (ignored, uses Piper)")
    input: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(default="minhkhang", description="Voice ID")
    response_format: str = Field(default="aac", description="Output format: aac, amr, mp3, m4a, wav, ogg")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch shift")


@app.post("/v1/audio/speech", tags=["OpenAI-Compatible"])
async def openai_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible TTS endpoint.
    Accepts the same request format as OpenAI's /v1/audio/speech API.
    """
    try:
        # Preprocess Vietnamese text
        processed_input = process_vietnamese_text(request.input)

        audio, sample_rate = tts_engine.synthesize(
            text=processed_input,
            voice=request.voice,
            speed=request.speed,
        )

        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="No audio generated")

        wav_bytes = tts_engine.audio_to_wav_bytes(audio, sample_rate)

        # Apply pitch
        if request.pitch != 1.0:
            wav_bytes = apply_pitch(wav_bytes, request.pitch, sample_rate)

        # Convert format
        out_bytes, media_type = convert_format(wav_bytes, request.response_format)
        return Response(content=out_bytes, media_type=media_type)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))



# === ASR (Speech-to-Text) Endpoints ===

from fastapi import UploadFile, File, Form
from asr_engine import ASREngine

ASR_MODELS_DIR = Path(os.environ.get("ASR_MODEL_DIR", str(Path(__file__).parent / "asr-models")))
asr_engine = ASREngine(str(ASR_MODELS_DIR))


@app.get("/asr/info", tags=["ASR"])
async def asr_info():
    """Get ASR engine status and supported languages."""
    return asr_engine.get_info()


@app.post("/asr", tags=["ASR"])
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, OGG, etc.)"),
    language: str = Form(default="vi", description="Language code (vi, en)"),
):
    """
    Transcribe audio file to text.
    Accepts any audio format (converted to WAV via ffmpeg).
    """
    if not asr_engine.is_ready:
        raise HTTPException(status_code=503, detail="ASR engine not ready")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        result = asr_engine.transcribe(audio_bytes, file.filename or "", language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions", tags=["OpenAI-Compatible"])
async def openai_transcriptions(
    file: UploadFile = File(..., description="Audio file"),
    model: str = Form(default="whisper-1", description="Model (ignored, uses whisper-tiny)"),
    language: str = Form(default="vi", description="Language code"),
    response_format: str = Form(default="json", description="Response format: json or text"),
):
    """
    OpenAI-compatible Whisper transcription endpoint.
    Compatible with OpenAI\'s /v1/audio/transcriptions API.
    """
    if not asr_engine.is_ready:
        raise HTTPException(status_code=503, detail="ASR engine not ready")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        result = asr_engine.transcribe(audio_bytes, file.filename or "", language)
        if response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === SRT Subtitle Export ===

@app.post("/asr/srt", tags=["ASR"])
async def transcribe_to_srt(
    file: UploadFile = File(..., description="Audio file"),
    language: str = Form(default="vi", description="Language code"),
):
    """
    Transcribe audio and return SRT subtitle file with timestamps.
    """
    if not asr_engine.is_ready:
        raise HTTPException(status_code=503, detail="ASR engine not ready")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        result = asr_engine.transcribe(audio_bytes, file.filename or "", language)
        text = result.get("text", "")
        duration = result.get("duration", 0)

        # Build SRT: split text into segments of ~10 words
        words = text.split()
        segments = []
        chunk_size = 10
        for i in range(0, len(words), chunk_size):
            segments.append(" ".join(words[i:i+chunk_size]))

        if not segments:
            segments = [text or "(no speech detected)"]

        seg_duration = duration / len(segments) if segments else duration
        srt_lines = []
        for idx, seg in enumerate(segments):
            start = idx * seg_duration
            end = (idx + 1) * seg_duration
            srt_lines.append(f"{idx+1}")
            srt_lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
            srt_lines.append(seg)
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)
        return Response(
            content=srt_content,
            media_type="text/srt",
            headers={"Content-Disposition": "attachment; filename=transcript.srt"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# === Text Chunking for Long Text ===

import io
import wave
import struct

def _chunk_text(text: str, max_len: int = 300) -> list:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_len:
        return [text]

    separators = ['. ', '! ', '? ', '; ', ', ', '\n']
    chunks = []
    current = ""

    sentences = []
    temp = text
    while temp:
        earliest_pos = len(temp)
        earliest_sep = ""
        for sep in separators:
            pos = temp.find(sep)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_sep = sep
        if earliest_sep:
            sentences.append(temp[:earliest_pos + len(earliest_sep)])
            temp = temp[earliest_pos + len(earliest_sep):]
        else:
            sentences.append(temp)
            break

    for sent in sentences:
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def _concatenate_wav(wav_list: list, sample_rate: int = 22050) -> bytes:
    """Concatenate multiple WAV byte arrays into one."""
    all_samples = bytearray()
    for wav_bytes in wav_list:
        bio = io.BytesIO(wav_bytes)
        with wave.open(bio, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            all_samples.extend(frames)

    output = io.BytesIO()
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(all_samples))
    return output.getvalue()


@app.post("/tts/long", tags=["TTS"])
async def synthesize_long_text(request: TTSRequest):
    """
    Synthesize long text with automatic chunking.
    Splits text at sentence boundaries, synthesizes each chunk, and concatenates audio.
    """
    try:
        processed_text = process_vietnamese_text(request.text)
        chunks = _chunk_text(processed_text)

        wav_parts = []
        for chunk in chunks:
            audio, sample_rate = tts_engine.synthesize(
                text=chunk, voice=request.voice, speed=request.speed,
                noise_scale=request.noise_scale, noise_w=request.noise_w,
            )
            if len(audio) > 0:
                wav_bytes = tts_engine.audio_to_wav_bytes(audio, sample_rate)
                if request.pitch != 1.0:
                    wav_bytes = apply_pitch(wav_bytes, request.pitch, sample_rate)
                wav_parts.append(wav_bytes)

        if not wav_parts:
            raise HTTPException(status_code=400, detail="No audio generated")

        if len(wav_parts) == 1:
            final = wav_parts[0]
        else:
            final = _concatenate_wav(wav_parts)

        return Response(
            content=final, media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Chunks": str(len(chunks)),
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
