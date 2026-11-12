# Voztudio

Built with Fish Audio.

Voztudio is a local multilingual voice-generation wrapper around the official
[Fish Speech](https://github.com/fishaudio/fish-speech) runtime. It accepts a
reference recording and a script, then exposes a local API and Gradio UI for
audio in English, Portuguese, Spanish, Hindi, Chinese, and French.

The language id is a label stored with the output. This Fish Speech build has
no language field, so the wrapper does not send it. `pt` does not distinguish
Brazilian Portuguese from European Portuguese. The accent follows the reference
recording, and the spoken language follows the script.

## Requirements

- Linux with Python 3.11 or 3.12, `uv`, Git, FFmpeg, and curl. `scripts/install.sh` uses `python3` when `python` is not on `PATH`.
- S2-Pro is the default model because it can run on CPU. GPU inference is strongly recommended. The installer selects the `cu128` PyTorch extra when `nvidia-smi` succeeds, and the `cpu` extra otherwise. Set `FISH_EXTRA` (`cpu`, `cu126`, `cu128`, or `cu129`) to override.
- A reference clip of 10–90 seconds (10–30 seconds is the recommended range), with its exact transcript. Near-silent audio is rejected.
- Drama 3 is not wired up. It needs an NVIDIA GPU with substantially more memory and has no CPU fallback.

## Install

```bash
scripts/install.sh
.venv/bin/hf download fishaudio/s2-pro --local-dir .cache/checkpoints/s2-pro
```

The installer pins the Fish Speech source to the revision recorded in the
script and installs it with `uv sync` so the upstream Protobuf override is
applied. `pyaudio` is skipped: the API server does not import it. Model
weights, reference audio, and generated outputs stay outside Git. Override
`FISH_SPEECH_REF`, `FISH_SPEECH_ROOT`, `FISH_EXTRA`, or `CHECKPOINT_ROOT`
when needed.

## Run locally

```bash
scripts/start.sh
```

The wrapper is at `http://127.0.0.1:7860`, its UI at
`http://127.0.0.1:7860/ui`, and the Fish Speech server remains internal at
`http://127.0.0.1:8080`. Copy `.env.example` to `.env` or export the same
variables. A value already set in the shell wins over `.env`. The file is
parsed as `KEY=VALUE` lines. It is not sourced.

`VOZTUDIO_HOST` and `FISH_HOST` must be loopback (`127.0.0.1`, `localhost`, or
`::1`) unless `VOZTUDIO_ALLOW_PUBLIC_BIND=1`. That flag prints a warning and
continues. `VOZTUDIO_DEVICE=auto` asks PyTorch for CUDA, then MPS, then CPU, and passes
that value as `--device`. `/health` keeps the requested value in `device` and
reports the process that actually ran in `resolved_device`. Fish Speech can
still move an S2-Pro process onto MPS when MPS is available, including when
`VOZTUDIO_DEVICE=cpu`. The log line is what `resolved_device` follows. The first load waits up to `FISH_START_TIMEOUT` seconds
(default 900) and requires both `config.json` and `codec.pth`.

Set `FISH_SPEECH_API_KEY` to require a bearer token on the model server. The
start script passes it to Fish Speech and does not print it.

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

Generate a script. A plain style note is sent as one inline tag, so `calm`
becomes `[calm]`. Tags you already wrote, such as `[whisper]`, are kept.
Scripts are split into roughly 200-character sentence chunks because this
S2-Pro server does not slice an untagged script itself.

```bash
curl -X POST http://127.0.0.1:7860/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello from my local voice studio.","language":"en","voice_id":"VOICE_ID","instructions":"calm"}'
```

Delete a voice or a generated file:

```bash
curl -X DELETE http://127.0.0.1:7860/voices/VOICE_ID
curl -X DELETE http://127.0.0.1:7860/outputs/OUTPUT_ID
```

Use `scripts/smoke_test.py --voice reference.wav --transcript '...'` to test all
six target languages after both servers are ready. The reference must already
satisfy the duration and level checks.

## License boundary

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
