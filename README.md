# Voztudio

Voztudio is a local multilingual voice-generation wrapper around the official
[Fish Speech](https://github.com/fishaudio/fish-speech) runtime. It accepts a
reference WAV and a script, then exposes a local API and Gradio UI for audio
in English, Brazilian Portuguese, Spanish, Hindi, Chinese, and French.

## Requirements

- Linux with Python 3.11 or 3.12, `uv`, Git, FFmpeg, and curl.
- S2-Pro is the default model because it has a CPU fallback. GPU inference is
  strongly recommended; Drama 3 requires substantially more NVIDIA VRAM.
- A clean mono WAV reference and its exact transcript. Fish Speech uses that
  transcript to prepare the voice prompt.

## Install

```bash
scripts/install.sh
.venv/bin/hf download fishaudio/s2-pro --local-dir .cache/checkpoints/s2-pro
```

The installer pins the Fish Speech source to the revision recorded in the
script and keeps the source, model weights, reference audio, and generated
outputs outside Git. Override `FISH_SPEECH_REF`, `FISH_SPEECH_ROOT`,
`FISH_EXTRA`, or `CHECKPOINT_ROOT` when needed.

## Run locally

```bash
scripts/start.sh
```

The S2-Pro CPU load can take several minutes on first start (`FISH_START_TIMEOUT` defaults to 900 seconds). The local script skips Fish Speech's long 1,023-token warm-up by default (`FISH_SKIP_WARMUP=0` restores it). The wrapper is at `http://127.0.0.1:7860`, its UI at
`http://127.0.0.1:7860/ui`, and the Fish Speech server remains internal at
`http://127.0.0.1:8080`. Configuration can be provided through `.env` or the
environment variables in `.env.example`.

## API

Register a reference voice:

```bash
curl -X POST http://127.0.0.1:7860/voices \
  -F 'audio=@reference.wav;type=audio/wav' \
  -F 'transcript=The exact words spoken in the reference recording.'
```

Generate a script:

```bash
curl -X POST http://127.0.0.1:7860/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello from my local voice studio.","language":"en","voice_id":"VOICE_ID"}'
```

Use `scripts/smoke_test.py --voice reference.wav --transcript '...'` to test all
six target languages after both servers are ready.

## License boundary

This project is a local wrapper. Fish Speech and its weights are governed by
the Fish Audio Community License; hosted self-service use requires a separate
license from Fish Audio. The wrapper does not remove or evade model provenance
signals. Only use a reference voice for which you have the necessary rights.
