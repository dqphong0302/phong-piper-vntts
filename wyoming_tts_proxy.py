"""
Wyoming TTS Proxy — proxies TTS requests to the existing FastAPI TTS server.
This avoids loading Piper models twice, saving ~255MB RAM.

Usage:
    python3 wyoming_tts_proxy.py --uri tcp://0.0.0.0:10200 --tts-url http://localhost/tts --voices-url http://localhost/voices
"""

import argparse
import asyncio
import io
import json
import logging
import struct
import wave
from typing import Optional
from urllib.request import urlopen, Request

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)


def fetch_voices(voices_url: str) -> list:
    """Fetch available voices from TTS API."""
    try:
        req = Request(voices_url)
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data if isinstance(data, list) else []
    except Exception as e:
        _LOGGER.error("Failed to fetch voices: %s", e)
        return []


def synthesize_via_api(tts_url: str, text: str, voice: str, speed: float = 1.0) -> bytes:
    """Call existing TTS API and return WAV bytes."""
    payload = json.dumps({"text": text, "voice": voice, "speed": speed}).encode("utf-8")
    req = Request(tts_url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


class TtsProxyHandler(AsyncEventHandler):
    """Wyoming handler that proxies to FastAPI TTS."""

    def __init__(self, tts_url: str, voices_url: str, default_voice: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tts_url = tts_url
        self.voices_url = voices_url
        self.default_voice = default_voice

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            voices = fetch_voices(self.voices_url)
            wyoming_voices = []
            for v in voices:
                vid = v.get("id", v.get("name", "unknown"))
                wyoming_voices.append(
                    TtsVoice(
                        name=vid,
                        description=vid,
                        languages=["vi", "vi_VN"],
                        attribution=Attribution(
                            name="Local Piper TTS",
                            url="http://localhost",
                        ),
                    )
                )

            info = Info(
                tts=[
                    TtsProgram(
                        name="local-piper",
                        description="Local Vietnamese Piper TTS (shared engine)",
                        attribution=Attribution(
                            name="Local Piper TTS",
                            url="http://localhost",
                        ),
                        voices=wyoming_voices,
                    )
                ]
            )
            await self.write_event(info.event())
            return True

        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            text = synth.text or ""
            voice = synth.voice or self.default_voice

            if not text.strip():
                return True

            _LOGGER.info("Synthesize: voice=%s, text=%s", voice, text[:50])

            try:
                wav_data = await asyncio.get_event_loop().run_in_executor(
                    None, synthesize_via_api, self.tts_url, text, voice
                )

                with io.BytesIO(wav_data) as wav_io:
                    with wave.open(wav_io, "rb") as wav_file:
                        rate = wav_file.getframerate()
                        width = wav_file.getsampwidth()
                        channels = wav_file.getnchannels()
                        frames = wav_file.readframes(wav_file.getnframes())

                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=channels).event()
                )

                # Send in chunks
                chunk_size = rate * width * channels  # 1 second chunks
                for i in range(0, len(frames), chunk_size):
                    chunk = frames[i : i + chunk_size]
                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event()
                    )

                await self.write_event(AudioStop().event())
                _LOGGER.info("Synthesis complete: %d bytes", len(frames))

            except Exception as e:
                _LOGGER.error("TTS synthesis failed: %s", e)

            return True

        return True


async def main():
    parser = argparse.ArgumentParser(description="Wyoming TTS Proxy")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="Wyoming server URI")
    parser.add_argument("--tts-url", default="http://localhost/tts", help="TTS API endpoint")
    parser.add_argument("--voices-url", default="http://localhost/voices", help="Voices API endpoint")
    parser.add_argument("--default-voice", default="lacphi", help="Default voice")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    _LOGGER.info(
        "Starting Wyoming TTS Proxy: uri=%s, tts=%s, voices=%s",
        args.uri, args.tts_url, args.voices_url,
    )

    server = AsyncTcpServer.from_uri(args.uri)
    await server.run(
        partial(
            TtsProxyHandler,
            args.tts_url,
            args.voices_url,
            args.default_voice,
        )
    )


if __name__ == "__main__":
    from functools import partial
    asyncio.run(main())
