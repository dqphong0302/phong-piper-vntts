# 🇻🇳 Phong Piper VNTTS

> **Vietnamese & English Text-to-Speech + Speech-to-Text API**  
> Powered by [Piper ONNX](https://github.com/rhasspy/piper) + [Sherpa-ONNX Whisper](https://github.com/k2-fsa/sherpa-onnx)

A self-hosted, production-ready Speech API with:
- **TTS**: 10+ Vietnamese voices + 3 English voices via Piper ONNX
- **ASR**: Whisper-based speech recognition (multi-language)
- **OpenAI-compatible** endpoints (`/v1/audio/speech`, `/v1/audio/transcriptions`)
- **Wyoming protocol** support for [Home Assistant](https://www.home-assistant.io/) integration
- **Vietnamese text preprocessing** — automatic number/abbreviation expansion
- Modern web UI with per-language voice galleries

## 📸 Preview

| Vietnamese TTS | English TTS |
|:-:|:-:|
| 🇻🇳 10 voices (5 nữ + 5 nam) | 🇺🇸 3 male voices |

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/phong-piper-vntts.git
cd phong-piper-vntts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy env and configure
cp .env.example .env
```

### 2. Download Models

Download Piper ONNX voice models and place them in `./models/`:

```bash
# Vietnamese voices (from NGHI-TTS)
# See: https://github.com/nghimestudio/nghitts
mkdir -p models
# Place .onnx + .onnx.json files in models/
```

### 3. Run

```bash
# Development
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production (systemd)
sudo cp tts.service /etc/systemd/system/
sudo systemctl enable tts && sudo systemctl start tts

# Docker
docker-compose up -d
```

## 📡 API Endpoints

### TTS (Text-to-Speech)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tts` | Convert text → WAV audio |
| `POST` | `/tts/long` | Long text TTS (auto-chunking) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS |
| `GET` | `/voices` | List available voices |

```bash
# Basic TTS
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào Việt Nam","voice":"lacphi","speed":1.0}' \
  -o speech.wav

# OpenAI-compatible
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"john","response_format":"mp3"}' \
  -o speech.mp3
```

### ASR (Speech-to-Text)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/asr` | Transcribe audio → text |
| `POST` | `/asr/srt` | Transcribe with SRT timestamps |
| `POST` | `/v1/audio/transcriptions` | OpenAI Whisper-compatible |
| `GET` | `/asr/info` | ASR engine info |

```bash
# Transcribe
curl -X POST http://localhost:8000/asr \
  -F "file=@audio.wav" -F "language=vi"

# SRT subtitles
curl -X POST http://localhost:8000/asr/srt \
  -F "file=@audio.wav" -F "language=vi" -o transcript.srt
```

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/` | Web UI |

## 🎙️ Available Voices

### Vietnamese (vi)

| Voice ID | Giọng | Mô tả |
|----------|-------|-------|
| `lacphi` | Nữ | Chững chạc, đĩnh đạc |
| `maiphuong` | Nữ | Trong trẻo, tươi sáng |
| `mytam2` | Nữ | Ca sĩ |
| `ngochuyen` | Nữ | Truyền cảm |
| `thanhphuong2` | Nữ | Nhẹ nhàng |
| `manhdung` | Nam | Rõ ràng, mạch lạc |
| `minhkhang` | Nam | Mạnh mẽ, tự tin |
| `minhquang` | Nam | Tự tin, chuyên nghiệp |
| `tranthanh3870` | Nam | Nghệ sĩ, hài hước |
| `vietthao3886` | Nam | MC, truyền hình |

### English (en)

| Voice ID | Gender | Description |
|----------|--------|-------------|
| `john` | Male | Clear, professional |
| `mattheo` | Male | Natural, expressive |
| `mattheo1` | Male | Warm, conversational |

## ⚙️ Configuration

All configuration is done via environment variables (`.env` file):

```ini
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# CORS
CORS_ORIGINS=*

# Models
MODELS_DIR=./models
ASR_MODEL_DIR=./asr-models
DEFAULT_VOICE=lacphi
ASR_MODEL=sherpa-onnx-whisper-medium
```

## 🏗️ Architecture

```
phong-piper-vntts/
├── app.py                  # FastAPI main application
├── tts_engine.py           # Piper ONNX TTS engine
├── asr_engine.py           # Sherpa-ONNX Whisper ASR engine
├── vietnamese_processor.py # Vietnamese text preprocessing
├── wyoming_tts_proxy.py    # Wyoming protocol proxy for HA
├── static/
│   └── index.html          # Web UI
├── models/                 # Piper .onnx voice models (gitignored)
├── asr-models/             # Whisper models (gitignored)
├── acronyms.csv            # Vietnamese acronym expansions
├── non-vietnamese-words.csv# Foreign word pronunciations
├── docker-compose.yml      # Docker deployment
├── Dockerfile              # Container build
├── tts.service             # systemd service unit
├── setup_lxc.sh            # LXC container setup script
├── requirements.txt        # Python dependencies
└── .env.example            # Environment template
```

## 🏠 Home Assistant Integration

This server supports the [Wyoming protocol](https://www.home-assistant.io/integrations/wyoming/), enabling direct integration with Home Assistant as a TTS provider:

```bash
# The Wyoming TTS proxy runs alongside the main API
# Configure in Home Assistant: Settings → Integrations → Wyoming
# Host: your-server-ip, Port: 10200
```

## 🙏 Credits

- **[NGHI-TTS](https://github.com/nghimestudio/nghitts)** — Vietnamese & English voice models
- **[Piper](https://github.com/rhasspy/piper)** — Fast ONNX neural TTS engine
- **[Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)** — Whisper ASR inference
- **[FastAPI](https://fastapi.tiangolo.com/)** — Web framework

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

*Built and hosted by **Phong Đặng** on a home server.*
