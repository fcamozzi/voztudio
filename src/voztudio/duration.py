"""Stop a chunk from talking past the script.

VoxCPM2's MLX ``generate`` loops until its stop head fires or ``max_tokens``
patches are reached. The published default is 2000 patches, and OpenBMB's
VoxCPM2 card lists a 6.25 Hz language-model token rate, so that default is
several minutes of audio. Qwen3-TTS-12Hz stops on the codec EOS token inside
a ``max_tokens`` loop whose own default is 2048 (about 170 seconds at 12 Hz).
mlx-audio's batched Qwen path already clamps that to ``max(75, len(text_tokens) * 6)``.

The character rate below is this wrapper's cap, not an upstream measurement.
It is deliberately slower than conversational speech so a slow reading still
fits, and still too short for a second invented sentence.
"""

from __future__ import annotations

import math

# OpenBMB lists VoxCPM2's LM token rate as 6.25 Hz. One generate() step is one patch.
VOX_PATCH_HZ = 6.25
# The checkpoint name and the MLX decoder are the 12 Hz codec.
QWEN_CODEC_HZ = 12.0
# mlx-audio's batched Qwen generate refuses to go under 75 codec tokens.
QWEN_MIN_TOKENS = 75

_CHARS_PER_SECOND = 11.0
_MARGIN = 1.7
_PAD_S = 1.0
_MIN_SECONDS = 2.5


def cap_seconds(text: str) -> float:
    """Upper bound on how long this chunk is allowed to sound."""
    spoken = max(1, len(text.strip()))
    expected = max(_MIN_SECONDS, spoken / _CHARS_PER_SECOND)
    return expected * _MARGIN + _PAD_S


def qwen_max_tokens(text: str) -> int:
    """Codec tokens for one Qwen3 chunk, including the EOS step."""
    allowed = max(QWEN_MIN_TOKENS, math.ceil(cap_seconds(text) * QWEN_CODEC_HZ))
    return int(allowed)


def vox_max_tokens(text: str) -> int:
    """Audio patches for one VoxCPM2 chunk. The stop head can still end sooner."""
    return max(8, math.ceil(cap_seconds(text) * VOX_PATCH_HZ))


def trim_to_duration(
    samples: tuple[float, ...] | list[float],
    sample_rate: int,
    max_seconds: float,
) -> tuple[tuple[float, ...], bool]:
    """Drop audio past ``max_seconds``, preferring a quiet point just before the cut."""
    if sample_rate <= 0 or max_seconds <= 0:
        return tuple(samples), False
    limit = int(max_seconds * sample_rate)
    if len(samples) <= limit:
        return tuple(samples), False
    window = max(1, int(0.02 * sample_rate))
    search = max(0, limit - int(0.35 * sample_rate))
    cut = limit
    quiet = 0.02
    index = search
    while index + window <= limit:
        rms = _rms(samples[index : index + window])
        if rms < quiet:
            cut = index
        index += window
    trimmed = [float(sample) for sample in samples[:cut]]
    fade = min(len(trimmed), max(1, int(0.015 * sample_rate)))
    if fade > 1:
        last = fade - 1
        for step in range(fade):
            trimmed[-fade + step] *= (last - step) / last
    return tuple(trimmed), True


def _rms(samples) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(samples))
