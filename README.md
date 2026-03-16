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

| Voice ID | Giọng | Vùng | Sample Rate |
|----------|-------|------|-------------|
| `valtec-nf` | ♀ Nữ Bắc | North Female | 24kHz |
| `valtec-sf` | ♀ Nữ Nam | South Female | 24kHz |
| `valtec-sm` | ♂ Nam Nam | South Male | 24kHz |
| `valtec-nm2` | ♂ Nam Bắc | North Male | 24kHz |

### 🇻🇳 Piper · Giọng Nữ (5)

| Voice ID | Tên | Mô tả | Sample Rate |
|----------|-----|-------|-------------|
| `lacphi` | Lạc Phi | Chững chạc, đĩnh đạc | 22.05kHz |
| `maiphuong` | Mai Phương | Trong trẻo, tươi sáng | 22.05kHz |
| `ngochuyen` | Ngọc Huyền | Truyền cảm | 22.05kHz |
| `thanhphuong2` | Thanh Phương | Nhẹ nhàng | 22.05kHz |

### 🇻🇳 Piper · Giọng Nam (4)

| Voice ID | Tên | Mô tả | Sample Rate |
|----------|-----|-------|-------------|
| `manhdung` | Mạnh Dũng | Rõ ràng, mạch lạc | 22.05kHz |
| `minhkhang` | Minh Khang | Mạnh mẽ, tự tin | 22.05kHz |
| `minhquang` | Minh Quang | Tự tin, chuyên nghiệp | 22.05kHz |
| `tranthanh3870` | Trấn Thành | Nghệ sĩ, hài hước | 22.05kHz |

### 🇺🇸 English (3)

| Voice ID | Description |
|----------|-------------|
| `john` | Clear, professional |
| `mattheo` | Natural, expressive |
| `mattheo1` | Warm, conversational |

---

## 🔊 Audio Processing Pipeline

### Piper voices

```
Text → Vietnamese Preprocessor → Piper ONNX model → float32 PCM → WAV/MP3/...
           (số, viết tắt,              (22.05kHz)
            dấu câu)
```

1. **Text Preprocessing** (`vietnamese_processor.py`): mở rộng số (1000 → "một nghìn"), viết tắt (TP.HCM → "Thành phố Hồ Chí Minh"), từ nước ngoài
2. **Piper Inference**: text → phoneme → VITS neural vocoder → audio waveform
3. **Post-processing**: pitch shift (ffmpeg asetrate), format convert

### Valtec ONNX voices

```
Text → Vietnamese Preprocessor → Phonemizer → Token IDs → ONNX Runtime → audio
           (chuẩn hóa)        (ViPhoneme)   (phone+tone      (VITS)      (24kHz)
                                              +language)
```

1. **Text Preprocessing**: chuẩn hóa tiếng Việt (số, viết tắt, dấu câu)
2. **Phonemization**: text → Vietnamese phonemes → tone IDs + language IDs
3. **Token Encoding**: phoneme → `cleaned_text_to_sequence()` → int64 tensors, blanks interspersed
4. **ONNX Inference**: gửi vào ONNX Runtime với inputs:

| Input | Shape | Mô tả |
|-------|-------|-------|
| `x` | `[1, seq_len]` | Phone ID sequence |
| `x_lengths` | `[1]` | Sequence length |
| `tone` | `[1, seq_len]` | Tone IDs |
| `language` | `[1, seq_len]` | Language IDs (VI=0) |
| `bert` | `[1, 1024, seq_len]` | BERT features (zeros — disabled) |
| `ja_bert` | `[1, 768, seq_len]` | JA-BERT features (zeros) |
| `noise_scale` | scalar | Variation randomness (0.667) |
| `length_scale` | scalar | Duration/speed control |
| `noise_scale_w` | scalar | Duration noise (0.8) |
| `sdp_ratio` | scalar | Stochastic duration ratio (0.0) |

5. **Output**: `[1, 1, audio_samples]` float32 waveform (24kHz)

---

## 🔧 ONNX Export Process

Valtec-TTS sử dụng kiến trúc **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech). Quá trình xuất ONNX:

### Từ PyTorch → ONNX

```bash
# Script: export_onnx.py
python export_onnx.py
```

**Các bước:**

1. **Load PyTorch model** từ `~/.cache/valtec_tts/models/vits-vietnamese/`
2. **Wrap model** trong `ValtecInferWrapper`:
   - Gắn cố định speaker embedding (1 file per speaker)
   - Chuyển tham số scalar (noise_scale, length_scale) thành input tensors
   - Gọi `model.infer()` thay vì `model.forward()`
3. **Export** bằng `torch.onnx.export()`:
   - Dynamic axes cho `x`, `tone`, `language` (chiều seq_len thay đổi)
   - Opset 14, float32
4. **Kết quả**: 1 file `.onnx` (~168MB) per speaker + 1 file `.onnx.json` config

### Download Models

ONNX models không lưu trong git (quá lớn). Tải về từ Google Drive:

📥 **[Download ONNX Models (Google Drive)](https://drive.google.com/drive/folders/1wAT-dDHECTblEjBsjdj-dOFfqfvyWVGT?usp=sharing)**

Đặt file `.onnx` + `.onnx.json` vào thư mục `models/`.

### Model Sizes

| Type | Per model | Total |
|------|-----------|-------|
| Valtec ONNX | ~168 MB | 672 MB (4 speakers) |
| Piper ONNX | ~61 MB | 671 MB (11 models) |

### Benchmark (4 CPU, 6GB RAM)

| Engine | Câu demo ~80 ký tự | So sánh |
|--------|---------------------|---------|
| Valtec PyTorch | 24.3s | baseline |
| Valtec PyTorch + `torch.compile` | 10.9s | 2.2x |
| **Valtec ONNX** | **2.3s** | **10.5x** |
| Piper ONNX | 1.6s | 15x |

---

## 📡 API

### TTS Parameters

| Param | Type | Range | Default | Mô tả |
|-------|------|-------|---------|-------|
| `text` / `input` | string | 1–5000 chars | — | Văn bản cần đọc |
| `voice` | string | — | `lacphi` | Voice ID |
| `speed` | float | 0.5–2.0 | 1.0 | Tốc độ đọc (2.0 = nhanh gấp đôi) |
| `pitch` | float | 0.5–2.0 | 1.0 | Cao độ giọng (ffmpeg asetrate) |
| `noise_scale` | float | 0.0–1.0 | — | Mức biến đổi ngẫu nhiên |
| `noise_w` | float | 0.0–1.0 | — | Độ rộng nhiễu duration |
| `response_format` | string | — | `aac` | Format đầu ra |

### Output Formats

| Format | Codec | Bitrate | Sample Rate |
|--------|-------|---------|-------------|
| `wav` | PCM 16-bit | lossless | 22.05/24 kHz |
| `mp3` | libmp3lame | 128k | 44.1 kHz |
| `m4a` | AAC | 128k | 44.1 kHz |
| `aac` | AAC | 64k | 44.1 kHz |
| `ogg` / `opus` | libopus | 64k | 48 kHz |
| `amr` | libopencore_amrnb | 12.2k | 8 kHz |

### Ví dụ

```bash
# Valtec voice, chậm lại, format mp3
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Xin chào","voice":"valtec-sf","speed":0.8,"response_format":"mp3"}' \
  -o speech.mp3

# Piper voice, giọng cao hơn
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Xin chào","voice":"lacphi","pitch":1.3}' \
  -o speech.wav

# Long text (auto-chunking tại dấu câu)
curl -X POST http://localhost:8000/tts/long \
  -H "Content-Type: application/json" \
  -d '{"text":"Đoạn văn rất dài...","voice":"minhkhang"}' \
  -o speech.wav
```

### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS |
| `POST` | `/tts` | TTS → audio (all formats) |
| `POST` | `/tts/long` | Long text TTS (auto-chunking) |
| `GET` | `/voices` | Danh sách voices |
| `POST` | `/v1/audio/transcriptions` | OpenAI Whisper-compatible ASR |
| `POST` | `/asr` | Transcribe audio |
| `POST` | `/asr/srt` | Transcribe → SRT subtitle |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

---

## ⚙️ Config

```ini
# .env
HOST=0.0.0.0
PORT=8000
MODELS_DIR=./models
DEFAULT_VOICE=lacphi
ASR_MODEL_DIR=./asr-models
ASR_MODEL=sherpa-onnx-whisper-medium
OMP_NUM_THREADS=4          # match CPU count
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
├── export_onnx.py          # Valtec PyTorch → ONNX export script
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
