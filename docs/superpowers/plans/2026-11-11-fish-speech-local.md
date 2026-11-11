# Fish Speech Local Voice Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Voztudio app that clones a user-provided voice reference and generates script audio with Fish Speech across English, Brazilian Portuguese, Spanish, Hindi, Chinese, and French.

**Architecture:** A thin Python wrapper owns local voice/output storage, request validation, and a stable API/UI boundary. A Fish Speech adapter owns model loading and inference, with a fake engine used by unit tests so tests do not download model weights. The local server binds to loopback and keeps model/audio caches outside Git.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, Gradio, pytest, soundfile/FFmpeg, PyTorch, official Fish Speech source and model release.

**Spec:** `docs/superpowers/specs/2026-11-11-fish-speech-local-design.md`

## Global Constraints

- The first version is local-only and binds to `127.0.0.1`.
- Target languages are `en`, `pt`, `es`, `hi`, `zh`, and `fr`.
- Reference audio, outputs, model weights, and credentials stay outside Git.
- The upstream Fish Speech revision and model identifier are explicit configuration.
- The app preserves upstream provenance/watermark behavior and provides no removal feature.
- The implementation must state the Fish Audio Community License boundary for hosted services.

## Review Focus

- Missing or invalid reference audio returns a clear 4xx error and never starts inference; test in `tests/test_voices.py`.
- Unsupported language IDs are rejected before model execution; test in `tests/test_requests.py`.
- Long scripts are chunked deterministically and outputs are concatenated in order; test in `tests/test_generation.py`.
- Model/download/device failures report actionable readiness details through `/health`; test in `tests/test_api.py`.
- Restarting the service with an existing data directory preserves registered voice metadata and generated outputs; test in `tests/test_storage.py`.

### Task 1: Scaffold the Python package and local configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/voztudio/__init__.py`
- Create: `src/voztudio/config.py`
- Create: `tests/test_config.py`
- Create: `README.md`

**Interfaces:**
- Produces `Settings` with `data_dir`, `cache_dir`, `fish_speech_root`, `model_id`, `device`, `host`, `port`, and the six-language tuple.
- Produces `get_settings() -> Settings`, reading environment variables with safe local defaults.

- [ ] **Step 1: Write failing configuration tests**

Test default language IDs, loopback host, and environment-variable overrides.

- [ ] **Step 2: Run `pytest tests/test_config.py -q` and verify failure**

Expected: import/attribute failures because the package does not exist yet.

- [ ] **Step 3: Implement package metadata, settings, ignored local directories, and README setup overview**

Use Python 3.11, `src` layout, and defaults under `./data` and `./.cache` while allowing absolute overrides.

- [ ] **Step 4: Run `pytest tests/test_config.py -q` and verify pass**

- [ ] **Step 5: Commit scaffold**

```bash
git add pyproject.toml .env.example .gitignore src tests README.md
git commit -m "chore: scaffold local voztudio runtime"
```

### Task 2: Implement voice storage and request/domain validation

**Files:**
- Create: `src/voztudio/domain.py`
- Create: `src/voztudio/storage.py`
- Create: `src/voztudio/text.py`
- Create: `tests/test_voices.py`
- Create: `tests/test_requests.py`
- Modify: `src/voztudio/config.py`

**Interfaces:**
- `VoiceRecord(id: str, audio_path: Path, transcript: str | None, sample_rate: int, created_at: datetime)`.
- `VoiceStore(root: Path).register(source: Path, transcript: str | None = None) -> VoiceRecord`.
- `VoiceStore.list() -> list[VoiceRecord]` and `VoiceStore.get(voice_id: str) -> VoiceRecord`.
- `validate_language(language: str) -> str` and `chunk_text(text: str, max_chars: int) -> list[str]`.

- [ ] **Step 1: Write failing tests**

Cover WAV validation, rejection of empty/non-audio input, persistence across a new `VoiceStore`, exact language set, and deterministic paragraph-aware chunking.

- [ ] **Step 2: Run `pytest tests/test_voices.py tests/test_requests.py -q` and verify failure**

- [ ] **Step 3: Implement storage and validation**

Copy accepted audio into a generated voice directory, normalize metadata without modifying the original upload, persist JSON metadata atomically, and keep paths below the configured data root.

- [ ] **Step 4: Run the focused tests and verify pass**

- [ ] **Step 5: Commit domain/storage layer**

```bash
git add src/voztudio/domain.py src/voztudio/storage.py src/voztudio/text.py src/voztudio/config.py tests
git commit -m "feat: add voice storage and generation validation"
```

### Task 3: Add Fish Speech inference adapter and generation service

**Files:**
- Create: `src/voztudio/engine.py`
- Create: `src/voztudio/generation.py`
- Create: `tests/test_generation.py`
- Modify: `src/voztudio/domain.py`

**Interfaces:**
- `GenerationRequest(text: str, language: str, voice_id: str, instructions: str | None = None, output_format: Literal["wav"] = "wav")`.
- `GeneratedAudio(id: str, audio_path: Path, sample_rate: int, duration_s: float, language: str, voice_id: str)`.
- `SpeechEngine.generate(text: str, language: str, reference: VoiceRecord, instructions: str | None = None) -> AudioBuffer`.
- `FishSpeechEngine.from_settings(settings: Settings) -> FishSpeechEngine` and `.generate(...) -> AudioBuffer`.
- `GenerationService(engine: SpeechEngine, store: VoiceStore, output_root: Path).generate(request: GenerationRequest) -> GeneratedAudio`.

- [ ] **Step 1: Write failing tests with a deterministic fake engine**

Assert chunk order, selected voice, language forwarding, output metadata, and actionable errors when the real engine cannot load.

- [ ] **Step 2: Run `pytest tests/test_generation.py -q` and verify failure**

- [ ] **Step 3: Implement chunked generation and output writing**

Use the fake-engine interface for tests. The real adapter must isolate upstream imports, load once, reuse a prepared voice prompt where supported, preserve Fish Speech controls, and make device/model errors explicit.

- [ ] **Step 4: Run focused tests and verify pass**

- [ ] **Step 5: Commit inference layer**

```bash
git add src/voztudio/engine.py src/voztudio/generation.py src/voztudio/domain.py tests/test_generation.py
git commit -m "feat: add Fish Speech generation service"
```

### Task 4: Expose the local API and browser UI

**Files:**
- Create: `src/voztudio/api.py`
- Create: `src/voztudio/ui.py`
- Create: `src/voztudio/app.py`
- Create: `tests/test_api.py`

**Interfaces:**
- `create_app(settings: Settings | None = None, engine: SpeechEngine | None = None) -> FastAPI`.
- `GET /health` returns device/model readiness and data-root status.
- `GET /voices` lists registered voices.
- `POST /voices` accepts multipart audio plus optional transcript and returns `VoiceRecord` JSON.
- `POST /generate` accepts `GenerationRequest` JSON and returns `GeneratedAudio` JSON.
- `GET /outputs/{output_id}` streams the generated WAV.
- `create_ui(service: GenerationService) -> gradio.Blocks`.

- [ ] **Step 1: Write failing API tests with the fake engine**

Cover health, voice registration, language rejection, generation response, and output download.

- [ ] **Step 2: Run `pytest tests/test_api.py -q` and verify failure**

- [ ] **Step 3: Implement FastAPI composition and Gradio UI**

Keep API and UI on the same service object, bind only to configured host/port, return structured error messages, and avoid exposing arbitrary filesystem paths.

- [ ] **Step 4: Run API tests and verify pass**

- [ ] **Step 5: Commit API/UI**

```bash
git add src/voztudio/api.py src/voztudio/ui.py src/voztudio/app.py tests/test_api.py
git commit -m "feat: add local API and Gradio interface"
```

### Task 5: Make installation, startup, and real-model validation repeatable

**Files:**
- Create: `scripts/install.sh`
- Create: `scripts/start.sh`
- Create: `scripts/smoke_test.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- `scripts/install.sh` installs wrapper dependencies, verifies FFmpeg/Python, and installs the pinned Fish Speech revision without disabling TLS/checksum verification.
- `scripts/start.sh` loads `.env`, checks model/cache paths, and runs `python -m voztudio.app`.
- `scripts/smoke_test.py --voice <path>` generates one short sentence per target language and exits nonzero on any failed language.

- [ ] **Step 1: Add script contract tests or dry-run checks**

Verify scripts use the repository root, honor overrides, fail on missing prerequisites, and never write model/audio artifacts into tracked paths.

- [ ] **Step 2: Implement scripts and operational documentation**

Document GPU/CPU expectations, first-run model download, reference-audio preparation, local URLs, Fish Speech license boundary, and the exact environment variables.

- [ ] **Step 3: Run the full unit/API suite**

```bash
pytest -q
```

Expected: all deterministic tests pass without downloading model weights.

- [ ] **Step 4: Run the real readiness checks**

Run `scripts/start.sh`, call `GET /health`, register a real reference WAV if available, and run `scripts/smoke_test.py --voice ...`. Record per-language results and distinguish unavailable GPU/model downloads from code failures.

- [ ] **Step 5: Verify restart persistence and clean tracked state**

Stop only the process started by the task, restart it, repeat health and output retrieval, then run `git status --short` and confirm only ignored local data/caches remain.

- [ ] **Step 6: Commit operational setup**

```bash
git add scripts README.md pyproject.toml
git commit -m "chore: add reproducible local setup and smoke test"
```
