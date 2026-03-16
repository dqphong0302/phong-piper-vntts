# 🇻🇳 Phong Piper VNTTS

> **Vietnamese & English Text-to-Speech + Speech-to-Text API**
> [Piper ONNX](https://github.com/rhasspy/piper) · [Valtec ONNX](https://github.com/tronghieuit/valtec-tts) · [Sherpa-ONNX Whisper](https://github.com/k2-fsa/sherpa-onnx)

Self-hosted Speech API — 15 voices, OpenAI-compatible, Home Assistant ready.

## 🚀 Quick Start

```bash
git clone https://github.com/dqphong0302/phong-piper-vntts.git
cd phong-piper-vntts

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 🎙️ Voices (15)

### 🎙️ Valtec · Nhấn nhá theo vùng miền (4)

Exported từ [tronghieuit/valtec-tts](https://github.com/tronghieuit/valtec-tts) sang ONNX. Chất lượng cao, nhấn nhá tự nhiên theo vùng miền.

| Voice ID | Giọng | Vùng |
|----------|-------|------|
| `valtec-nf` | ♀ Nữ Bắc | North Female |
| `valtec-sf` | ♀ Nữ Nam | South Female |
| `valtec-sm` | ♂ Nam Nam | South Male |
| `valtec-nm2` | ♂ Nam Bắc | North Male |

### 🇻🇳 Piper · Giọng Nữ (5)

| Voice ID | Tên | Mô tả |
|----------|-----|-------|
| `lacphi` | Lạc Phi | Chững chạc, đĩnh đạc |
| `maiphuong` | Mai Phương | Trong trẻo, tươi sáng |
| `ngochuyen` | Ngọc Huyền | Truyền cảm |
| `thanhphuong2` | Thanh Phương | Nhẹ nhàng |

### 🇻🇳 Piper · Giọng Nam (4)

| Voice ID | Tên | Mô tả |
|----------|-----|-------|
| `manhdung` | Mạnh Dũng | Rõ ràng, mạch lạc |
| `minhkhang` | Minh Khang | Mạnh mẽ, tự tin |
| `minhquang` | Minh Quang | Tự tin, chuyên nghiệp |
| `tranthanh3870` | Trấn Thành | Nghệ sĩ, hài hước |

### 🇺🇸 English (3)

| Voice ID | Description |
|----------|-------------|
| `john` | Clear, professional |
| `mattheo` | Natural, expressive |
| `mattheo1` | Warm, conversational |

## 📡 API

### TTS

```bash
# OpenAI-compatible
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Xin chào Việt Nam","voice":"valtec-sf"}' \
  -o speech.wav

# Long text (auto-chunking)
curl -X POST http://localhost:8000/tts/long \
  -H "Content-Type: application/json" \
  -d '{"text":"Đoạn văn dài...","voice":"lacphi"}' \
  -o speech.wav

# Output formats: wav, mp3, m4a, ogg, opus, aac
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"input":"Hello","voice":"john","response_format":"mp3"}' \
  -o speech.mp3
```

### ASR (Speech-to-Text)

```bash
# Transcribe
curl -X POST http://localhost:8000/asr -F "file=@audio.wav" -F "language=vi"

# OpenAI Whisper-compatible
curl -X POST http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav"

# SRT subtitles
curl -X POST http://localhost:8000/asr/srt -F "file=@audio.wav" -o transcript.srt
```

### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS |
| `POST` | `/tts` | TTS → WAV |
| `POST` | `/tts/long` | Long text TTS (auto-chunking) |
| `GET` | `/voices` | Danh sách voices |
| `POST` | `/v1/audio/transcriptions` | OpenAI Whisper-compatible ASR |
| `POST` | `/asr` | Transcribe audio |
| `POST` | `/asr/srt` | Transcribe → SRT |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

## ⚙️ Config

```ini
# .env
HOST=0.0.0.0
PORT=8000
MODELS_DIR=./models
DEFAULT_VOICE=lacphi
ASR_MODEL_DIR=./asr-models
ASR_MODEL=sherpa-onnx-whisper-medium
```

## 🏗️ Architecture

```
phong-piper-vntts/
├── app.py                  # FastAPI application
├── tts_engine.py           # Piper + Valtec voice routing
├── valtec_onnx_engine.py   # Valtec ONNX inference engine
├── asr_engine.py           # Sherpa-ONNX Whisper ASR
├── vietnamese_processor.py # Vietnamese text preprocessing
├── wyoming_tts_proxy.py    # Wyoming protocol (Home Assistant)
├── static/index.html       # Web UI
├── models/                 # ONNX voice models (gitignored)
│   ├── *.onnx              # Piper models (~61MB each)
│   ├── *.onnx.json         # Model configs
│   └── valtec-*.onnx       # Valtec models (~168MB each)
├── tts.service             # systemd service
├── docker-compose.yml      # Docker deployment
└── setup_lxc.sh            # LXC container setup
```

## 🏠 Home Assistant

Wyoming protocol support — add as TTS provider:

```
Settings → Integrations → Wyoming → Host: your-ip, Port: 10200
```

## 🙏 Credits

- **[NGHI-TTS](https://github.com/nghimestudio/nghitts)** — Vietnamese & English Piper voice models
- **[Valtec-TTS](https://github.com/tronghieuit/valtec-tts)** — Vietnamese multi-speaker VITS model (ONNX exported)
- **[Piper](https://github.com/rhasspy/piper)** — ONNX neural TTS engine
- **[Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)** — Whisper ASR
- **[FastAPI](https://fastapi.tiangolo.com/)** — Web framework

---

*Built by **Phong Đặng** · Hosted on Proxmox home server*
