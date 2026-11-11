# Voztudio local voice generation with Fish Speech

## Intent

Turn the empty `fcamozzi/voztudio` repository into a local, reproducible tool for generating narrated audio from a script with the user's own reference voice. The first useful version must cover English, Brazilian Portuguese, Spanish, Hindi, Chinese, and French without requiring voice fine-tuning.

## Goals

- Run Fish Speech locally with automatic GPU detection and a documented CPU fallback.
- Import and validate a short reference recording, keeping audio and model weights outside Git.
- Generate WAV audio from text with an explicit language, optional Fish Speech style tags, and a reusable voice prompt.
- Offer a small local HTTP API and browser UI for selecting a reference voice, entering a script, choosing language, and downloading the result.
- Store generated files in an ignored local data directory and provide health/readiness checks.
- Keep the upstream Fish Speech checkout and model version explicit so setup can be repeated.

## Non-goals for the first version

- Public hosting, accounts, billing, or multi-user access.
- Training or fine-tuning a custom model.
- A cloud API or self-service hosted voice product.
- Automatic consent or identity verification workflows.
- Committing reference audio, generated audio, model weights, or credentials.

## Approaches considered

1. **XTTS-v2 wrapper:** simple and covers the requested languages, but the upstream project is substantially less active and has fewer current controls for long-form, editing, and expressive generation.
2. **Fish Speech local wrapper (chosen):** current Drama 3/S2-Pro capabilities match the target product most closely: short-reference cloning, multilingual generation, long-form output, streaming, emotion tags, and editing. It requires a local runtime and the Fish Audio Community License; this design deliberately keeps the tool local and does not expose a hosted service.
3. **Qwen3-TTS wrapper:** Apache-2.0 and strong cloning, but the official language list omits Hindi, so it would require a second model or fallback path.

## Proposed design

- **Runtime:** Python 3.11, PyTorch, FFmpeg, and the official `fishaudio/fish-speech` source/model release. Detect CUDA at startup; use the smallest supported local model when GPU memory is constrained and report the selected device.
- **Project wrapper:** a thin `voztudio` package rather than a fork of Fish Speech. Keep the upstream code in an external/cache directory or a pinned editable dependency. Pin the upstream revision and model identifiers in configuration.
- **API:** local FastAPI service bound to `127.0.0.1`. Endpoints: `GET /health`, `GET /voices`, `POST /voices`, and `POST /generate`. Generation accepts `voice_id`, `text`, `language`, optional `instructions`, and output format. Return a job/result path with clear errors for missing models, invalid audio, or unsupported language.
- **UI:** a minimal local Gradio page on top of the same service: reference voice upload/selection, language selector, script editor, optional style tags, generate button, audio preview, and download. The UI is local-only and has no authentication in this milestone.
- **Audio flow:** normalize a reference clip to mono WAV at the model's expected sample rate, preserve the original file, create/reuse a model-specific voice prompt, synthesize in bounded chunks for long scripts, concatenate with FFmpeg, and write a timestamped output plus a JSON metadata sidecar.
- **Languages:** expose `en`, `pt`, `es`, `hi`, `zh`, and `fr` in the UI/API. Keep an explicit language field even when Fish Speech can infer it, so tests and reproducibility do not depend on inference.
- **Local data:** use `data/voices`, `data/outputs`, and `data/cache` with a checked-in `.gitignore`. Configuration comes from `.env.example` and environment variables; no secret values go into the repository.
- **Verification:** provide a smoke command that loads the model, synthesizes one short sentence per target language from a supplied reference, verifies non-empty WAV output and sample rate, and reports per-language status. Add an API health check and a restartable start command.

## Constraints and risks

- GPU VRAM and disk space determine whether Drama 3 or S2-Pro and which inference path are practical; setup must fail clearly rather than silently downloading incompatible weights.
- Fish Speech's Community License permits local/personal use but restricts hosted self-service use and may require a separate commercial license; the README must state this boundary.
- The reference voice must be supplied by the user with the necessary rights. The application will preserve upstream watermarks/provenance and will not include removal or evasion features.
- Upstream model downloads are large and must be cached outside Git. A first run may take several minutes.

## Definition of done for implementation

A fresh environment can run the documented install and start commands, open the local UI, register one reference WAV, generate successful audio in all six target languages, pass the smoke/readiness checks, and leave the repository clean apart from ignored local data and caches.
