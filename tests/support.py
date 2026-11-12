from __future__ import annotations

import array
import wave
from pathlib import Path


def write_reference(
    path: Path,
    seconds: float = 12.0,
    rate: int = 16000,
    amplitude: float = 0.2,
    channels: int = 1,
    constant: float | None = None,
) -> Path:
    """16-bit PCM WAV. A square wave by default, or a constant level."""
    frames = max(1, int(round(seconds * rate)))
    if constant is None:
        amplitude = max(-1.0, min(1.0, amplitude))
        level = int(amplitude * 32767)
        mono = array.array(
            "h",
            (level if (index // 8) % 2 == 0 else -level for index in range(frames)),
        )
    else:
        constant = max(-1.0, min(1.0, constant))
        level = int(constant * 32767)
        mono = array.array("h", (level for _ in range(frames)))
    payload = array.array("h")
    for sample in mono:
        payload.extend([sample] * channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(payload.tobytes())
    return path
