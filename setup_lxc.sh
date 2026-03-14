#!/bin/bash
# Deploy Vietnamese TTS API to Linux LXC
# Run this script on your LXC container

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}[1/5] Installing system dependencies...${NC}"
apt-get update
apt-get install -y python3 python3-venv python3-pip espeak-ng git

echo -e "${GREEN}[2/5] Creating TTS directory...${NC}"
mkdir -p /opt/tts
cd /opt/tts

echo -e "${GREEN}[3/5] Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}[4/5] Installing Python packages...${NC}"
pip install --upgrade pip
pip install fastapi uvicorn onnxruntime numpy piper-tts scipy python-multipart

echo -e "${GREEN}[5/5] Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Copy your files to /opt/tts/"
echo "   - app.py"
echo "   - tts_engine.py"
echo "   - static/index.html"
echo "   - models/*.onnx and models/*.onnx.json"
echo ""
echo "2. Start the server:"
echo "   cd /opt/tts && source venv/bin/activate"
echo "   uvicorn app:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. Or install as systemd service:"
echo "   cp tts.service /etc/systemd/system/"
echo "   systemctl enable tts && systemctl start tts"
