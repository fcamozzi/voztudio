"""In-process model loading.

The call signatures below were taken from upstream source. MLX is imported
only when that runtime is selected. On Linux and CUDA the official PyTorch
packages are the fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from voztudio.models import checkpoint_name


def mlx_weights(path: Path) -> bool:
    """True when a snapshot is an mlx-audio quantized conversion."""
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    quant = data.get("quantization")
    if not isinstance(quant, dict):
        quant = data.get("quantization_config")
    return isinstance(quant, dict) and quant.get("mode") == "affine"


def model_source(
    backend: str,
    runtime: str,
    model_id: str,
    cache_dir: Path,
    model_dir: Path | None = None,
) -> str:
    """Local snapshot when it matches the runtime, otherwise the hub id."""
    if model_dir is not None:
        return str(model_dir)
    local = cache_dir / "checkpoints" / checkpoint_name(backend)
    if (local / "config.json").is_file():
        quantized = mlx_weights(local)
        if runtime == "mlx" or not quantized:
            return str(local)
    return model_id


def torch_device(requested: str) -> str:
    import torch

    choice = (requested or "auto").strip() or "auto"
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_mlx(source: str):
    """``mlx_audio.tts.utils.load_model`` accepts a hub id or a local directory."""
    from mlx_audio.tts.utils import load_model

    return load_model(source)


def load_qwen_torch(source: str, device: str):
    """Official ``Qwen3TTSModel.from_pretrained`` kwargs.

    The upstream example uses ``device_map``, ``dtype``, and
    ``attn_implementation``. FlashAttention 2 is CUDA-only; other devices
    use SDPA. CPU uses float32.
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    if device.startswith("cuda"):
        dtype = torch.bfloat16
        device_map = device if ":" in device else "cuda:0"
        attn = "sdpa"
        try:
            import flash_attn  # noqa: F401

            attn = "flash_attention_2"
        except Exception:
            attn = "sdpa"
    elif device == "mps":
        dtype = torch.float16
        device_map = "mps"
        attn = "sdpa"
    else:
        dtype = torch.float32
        device_map = "cpu"
        attn = "sdpa"
    try:
        return Qwen3TTSModel.from_pretrained(
            source,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn,
        )
    except Exception:
        if attn == "sdpa":
            raise
        return Qwen3TTSModel.from_pretrained(
            source,
            device_map=device_map,
            dtype=dtype,
            attn_implementation="sdpa",
        )


def load_vox_torch(source: str, device: str):
    """``VoxCPM.from_pretrained(..., load_denoiser=False, device=)``.

    ``optimize=False`` avoids ``torch.compile`` on CPU and MPS. The upstream
    quick start documents that flag for those platforms.
    """
    from voxcpm import VoxCPM

    kwargs = {"load_denoiser": False, "device": device}
    if not device.startswith("cuda"):
        kwargs["optimize"] = False
    try:
        return VoxCPM.from_pretrained(source, **kwargs)
    except TypeError:
        kwargs.pop("optimize", None)
        return VoxCPM.from_pretrained(source, **kwargs)


def as_floats(audio) -> tuple[float, ...]:
    """Flatten an MLX array, a torch tensor, or a numpy waveform."""
    if audio is None:
        return ()
    if hasattr(audio, "detach") and hasattr(audio, "cpu"):
        audio = audio.detach().cpu().float().numpy()
    if hasattr(audio, "tolist"):
        audio = audio.tolist()
    flat: list[float] = []

    def walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
        else:
            flat.append(float(item))

    walk(audio)
    return tuple(flat)


def samples_from_mlx(results) -> tuple[tuple[float, ...], int]:
    chunks: list[tuple[float, ...]] = []
    rate = 0
    for result in results:
        audio = getattr(result, "audio", result)
        found = getattr(result, "sample_rate", None)
        if found:
            rate = int(found)
        chunks.append(as_floats(audio))
    merged = tuple(sample for chunk in chunks for sample in chunk)
    if not merged or rate <= 0:
        raise RuntimeError("The speech model returned no audio")
    return merged, rate

