from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


SAMPLES = {
    "en": "This is a short voice generation test.",
    "pt": "Este é um teste curto de geração de voz.",
    "es": "Esta es una prueba corta de generación de voz.",
    "hi": "यह आवाज़ निर्माण का एक छोटा परीक्षण है।",
    "zh": "这是一个简短的语音生成测试。",
    "fr": "Ceci est un court test de génération vocale.",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--transcript", required=True)
    args = parser.parse_args()
    timeout = httpx.Timeout(900.0, connect=10.0)
    with httpx.Client(base_url=args.url, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        voice = client.post(
            "/voices",
            files={"audio": (args.voice.name, args.voice.read_bytes(), "audio/wav")},
            data={"transcript": args.transcript, "consent": "true"},
        )
        voice.raise_for_status()
        voice_id = voice.json()["id"]
        results = {}
        for language, text in SAMPLES.items():
            response = client.post(
                "/generate",
                json={"text": text, "language": language, "voice_id": voice_id},
            )
            results[language] = response.status_code == 200
            response.raise_for_status()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
