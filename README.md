# Voztudio

Voztudio is a local wrapper for cloning the owner's own voice. It keeps a
reference recording, a Gradio UI, and a loopback API. The speech backend is
pluggable.

On an Apple Silicon Mac the default backend is [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
1.7B-Base, loaded in-process through [mlx-audio](https://github.com/Blaizzy/mlx-audio)
8-bit weights. A 16 GB M1 does not meet Fish Speech S2-Pro's published GPU
requirement, so that server is no longer started unless you ask for it.

`pt` is a label stored with the output. None of the backends separate
Brazilian Portuguese from European Portuguese. The accent follows the
reference recording, and the spoken language follows the script.

## Backends

| `VOZTUDIO_BACKEND` | Role | Apple Silicon weights | Linux / CUDA weights |
| --- | --- | --- | --- |
| `qwen3` (default) | Qwen3-TTS 1.7B-Base voice clone | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `qwen3-fast` | Qwen3-TTS 0.6B-Base voice clone | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| `voxcpm2` | VoxCPM2 controllable clone, with style | `mlx-community/VoxCPM2-8bit` | `openbmb/VoxCPM2` |
| `fish` | Fish Speech S2-Pro HTTP server | not the default; separate venv | same, separate venv |

`VOZTUDIO_RUNTIME` picks `mlx` or `pytorch`. Leave it empty: macOS arm64 uses
MLX, and every other machine uses the official PyTorch package. If MLX is
selected by default and `mlx-audio` is not installed, the process falls back
to the PyTorch repo and says so in `/health`. Setting `VOZTUDIO_RUNTIME=mlx`
explicitly does not fall back.

The UI shows the running backend. Changing the dropdown does not load another
model. Restart with the environment variable.

## Requirements

- Python 3.11 or 3.12, `uv`, Git, FFmpeg, and curl. `scripts/install.sh` uses `python3` when `python` is not on `PATH`.
- macOS system bash 3.2 is enough for the scripts. They do not use Bash 4 arrays or `mapfile`.
- A reference clip of 10–90 seconds, with its exact transcript. Near-silent audio is rejected. 10–30 seconds is the recommended upload. Synthesis sends a shorter prefix; see below.
- Qwen3-TTS Base does not list Hindi. The `hi` label is still stored, and that backend synthesizes with `Auto`. VoxCPM2's published language list includes Hindi and Portuguese.

## Install and run on a 16 GB M1

Default, Qwen3-TTS 1.7B-Base 8-bit:

```bash
scripts/install.sh
scripts/start.sh
```

Faster and smaller, Qwen3-TTS 0.6B-Base 8-bit:

```bash
VOZTUDIO_BACKEND=qwen3-fast scripts/install.sh
VOZTUDIO_BACKEND=qwen3-fast scripts/start.sh
```

VoxCPM2 8-bit, when you want style control:

```bash
VOZTUDIO_BACKEND=voxcpm2 scripts/install.sh
VOZTUDIO_BACKEND=voxcpm2 scripts/start.sh
```

The wrapper listens on `http://127.0.0.1:7860`. The UI is at
`http://127.0.0.1:7860/ui`. Copy `.env.example` to `.env` or export the same
variables. A value already set in the shell wins over `.env`. The file is
parsed as `KEY=VALUE` lines. It is not sourced.

`scripts/install.sh` downloads the selected checkpoint with `hf download`
into `.cache/checkpoints/`. Set `VOZTUDIO_SKIP_DOWNLOAD=1` to install
packages without the snapshot. `VOZTUDIO_MODEL_ID` replaces the repo id.
`VOZTUDIO_MODEL_DIR` points at a snapshot you already have.

The Mac install does not clone Fish Speech and does not pin NumPy to 1.26.
NumPy stays on `>=2.0.2`, which is what current Gradio and pandas need.

## Fish Speech on a GPU machine

S2-Pro stays available. It is installed only when you select it, and only
inside `.venv-fish`, so its dependency solve cannot downgrade the wrapper's
NumPy again.

```bash
VOZTUDIO_BACKEND=fish scripts/install.sh
VOZTUDIO_BACKEND=fish scripts/start.sh
```

The model server stays on `http://127.0.0.1:8080`. The installer pins the
Fish Speech source to the revision recorded in `scripts/install.sh` and
installs it with `uv sync` so the upstream Protobuf override is applied.
`pyaudio` is skipped. Override `FISH_SPEECH_REF`, `FISH_SPEECH_ROOT`,
`FISH_VENV_DIR`, `FISH_EXTRA`, or `CHECKPOINT_ROOT` when needed. `FISH_EXTRA`
is `cpu`, `cu126`, `cu128`, or `cu129`. The installer selects `cu128` when
`nvidia-smi` succeeds, and `cpu` otherwise.

`VOZTUDIO_HOST` and, for the fish backend, `FISH_HOST` must be loopback
(`127.0.0.1`, `localhost`, or `::1`) unless `VOZTUDIO_ALLOW_PUBLIC_BIND=1`.
That flag prints a warning and continues. The first Fish load waits up to
`FISH_START_TIMEOUT` seconds (default 900) and requires both `config.json`
and `codec.pth`.

Set `FISH_SPEECH_API_KEY` to require a bearer token on the model server. The
start script passes it through and does not print it.

`VOZTUDIO_DEVICE=auto` asks PyTorch for CUDA, then MPS, then CPU. `/health`
keeps the requested value in `device` and reports the device that actually
ran in `resolved_device`. For Fish, that value follows the server log. Fish
Speech can still move an S2-Pro process onto MPS when MPS is available.

On Linux or CUDA, `qwen3`, `qwen3-fast`, and `voxcpm2` use the official
PyTorch packages (`qwen-tts` and `voxcpm`) in the wrapper venv. MLX is an
Apple Silicon runtime. `qwen-tts` pins `transformers==4.57.3`, and that
release requires `huggingface_hub<1`. mlx-audio requires `huggingface_hub>=1`.
The installer keeps those solves apart: a Mac install does not install
`qwen-tts`, and a PyTorch install does not install `mlx-audio`. The
`qwen-tts` import warns when the `sox` binary is missing. Install it with
`brew install sox` on macOS, or the distro package on Linux, before using
that runtime.

## API

Register a reference voice. `consent` must be `true`, `1`, `yes`, or `on`.

```bash
curl -X POST http://127.0.0.1:7860/voices \
  -F 'audio=@reference.wav;type=audio/wav' \
  -F 'transcript=The exact words spoken in the reference recording.' \
  -F 'consent=true'
```

WAV, MP3, FLAC, Ogg, M4A, AAC, Opus, WebM, MP4, and AIFF are accepted. Other
files are decoded with FFmpeg to mono 16-bit PCM. Uploads larger than
`VOZTUDIO_MAX_UPLOAD_BYTES` (default 25 MiB) are rejected.

Generate a script. Scripts are split into roughly 200-character sentence
chunks and joined with a 20 ms crossfade. The reference prompt is cached for
the life of the process: Fish Speech still sends `use_memory_cache=on`, and
the Qwen3-TTS PyTorch path reuses `create_voice_clone_prompt`.

```bash
curl -X POST http://127.0.0.1:7860/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Olá, aqui é um teste da voz local.","language":"pt","voice_id":"VOICE_ID","instructions":"calm"}'
```

The JSON response includes `warnings`. Qwen3-TTS Base has no style channel,
so a style note or a `[tag]` is removed from the script and the warning says
so. An empty `warnings` list means nothing was dropped.

Delete a voice or a generated file:

```bash
curl -X DELETE http://127.0.0.1:7860/voices/VOICE_ID
curl -X DELETE http://127.0.0.1:7860/outputs/OUTPUT_ID
```

`/health` reports `backend`, `runtime`, `model_id`, `device`, and
`resolved_device`.

Use `scripts/smoke_test.py --voice reference.wav --transcript '...'` to test
all six target languages after the server is ready. The reference must
already satisfy the duration and level checks.

## Reference length

Uploads stay in the 10–90 second window. Synthesis does not send that whole
file:

| Backend | Prefix sent | Why |
| --- | --- | --- |
| Qwen3-TTS Base | first 15 s | The model card calls the Base checkpoints a 3-second rapid clone. Fifteen seconds is this wrapper's cap so a 90 s upload does not become the prompt. |
| VoxCPM2 | first 20 s | The VoxCPM usage guide calls 5–30 seconds a practical reference range. |
| Fish Speech | first 30 s | This project's existing recommended range is 10–30 seconds. |

The stored transcript is not rewritten. If the file was longer than the cap,
the warning says the transcript still describes the full upload. A transcript
of the kept prefix clones more accurately, especially for Qwen3-TTS, which
aligns the transcript with the reference audio.

## Style

Fish Speech still turns a plain note into one inline tag. `calm` is sent as
`[calm]`. Tags you already wrote, such as `[whisper]`, are kept.

Qwen3-TTS Base voice clone does not take those tags. The official model table
leaves instruction control empty for both Base sizes, and the MLX `generate`
path for a Base checkpoint does not apply `instruct`. The wrapper drops the
tags and returns a warning. Do not pass a speaker name either: mlx-audio
rejects `voice` together with `ref_audio` and `ref_text`.

VoxCPM2 controllable cloning reads one parenthetical instruction at the start
of the text, for example `(slightly faster, cheerful tone)This is a cloned voice`.
The MLX API takes the same words as `instruct` and adds the parentheses
itself. Known Fish-style tags are mapped to Brazilian Portuguese:

| Tag | VoxCPM2 instruction |
| --- | --- |
| `[whisper]`, `[whispering]` | `(sussurrando)` |
| `[soft]` | `(voz suave)` |
| `[calm]` | `(tom calmo)` |
| `[slow]` | `(mais devagar)` |
| `[fast]`, `[quickly]` | `(mais rápido)` |
| `[excited]` | `(animado)` |
| `[happy]` | `(tom alegre)` |
| `[sad]` | `(tom triste)` |
| `[angry]` | `(tom irritado)` |
| `[shout]`, `[shouting]`, `[loud]` | `(em voz alta)` |
| `[serious]` | `(tom sério)` |
| `[laugh]`, `[laughing]` | `(rindo)` |
| `[surprised]` | `(surpreso)` |
| `[breathy]` | `(voz soprosa)` |
| `[gentle]` | `(tom gentil)` |

A plain note that is already in the table is mapped too, so `calm` becomes
`(tom calmo)`. Several known notes separated by commas are combined:
`calm, slow` becomes `(tom calmo, mais devagar)`. A `[tag]` that is not in
the table is still sent, and the warning quotes it. A parenthetical you
already wrote, such as `(sussurrando)`, is kept.

This is controllable cloning (`reference_wav_path` / `ref_audio`), not
VoxCPM2 continuation. Continuation disables the control instruction.

## Memory and speed

These figures are copied from upstream docs. This repository does not add
its own Apple Silicon benchmark.

- Fish Speech's install guide says S2-Pro needs at least 24 GB of GPU memory and can run on an NVIDIA GPU or on CPU. A 16 GB M1 is below that requirement. Source: [Fish Audio installation](https://speech.fish.audio/install/).
- The Qwen3-TTS readme describes the 1.7B and 0.6B Base models as 3-second rapid voice clone, with Portuguese in the language list and Hindi absent. Source: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).
- The Hugging Face snapshot `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` is listed at about 3.1 GB, and its `model.safetensors` file is 2.42 GB. That is disk size, not unified-memory use while generating.
- mlx-audio's Qwen3-TTS readme publishes one memory table, for a 6-bit CustomVoice model on a short prompt, not for 8-bit Base on an M1: batch 1 is 20.8 tokens/s, 1.67× throughput, 84.8 ms time to first audio, and 3.88 GB. Source: [mlx-audio Qwen3-TTS readme](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/README.md).
- mlx-audio lists VoxCPM2 snapshot sizes as 4.96 GB for bf16, 3.23 GB for 8-bit, and 2.30 GB for 4-bit, at 48 kHz. The default here is the 8-bit snapshot. Source: [mlx-audio VoxCPM2 docs](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/tts/voxcpm2.md).
- OpenBMB lists VoxCPM2 as a 2B model, 48 kHz, about 8 GB of VRAM, and about 0.30 real-time factor on an RTX 4090 (about 0.13 with Nano-vLLM on the same GPU). Those speed numbers are not M1 measurements. The same page lists 30 languages, including Portuguese and Hindi. Source: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM).

Leave headroom for macOS. The 8-bit checkpoints are the ones selected so the
weights themselves are smaller than the bf16 Fish server that was swapping.

## License boundary

Qwen3-TTS is Apache-2.0. The license file in [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) is the Apache License 2.0, copyright 2026 Alibaba Cloud.

VoxCPM2 is Apache-2.0. mlx-audio and the [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) repository both say so.

The Fish Speech source pinned by `scripts/install.sh` is under the Fish Audio
Community License. That license allows research, non-commercial use, and
commercial use under its revenue threshold. Offering a hosted self-service
text-to-speech or voice-cloning service still needs a separate license from
Fish Audio.

The `fishaudio/s2-pro` weights are under the Fish Audio Research License.
Research and non-commercial use are allowed. Commercial use of those weights
needs a separate license from Fish Audio.

This wrapper does not remove or evade model provenance signals. Only register
a reference voice you have the rights to use. The API and the UI ask for that
confirmation before storing a voice.
