# Voztudio

Local multilingual voice generation built around the official Fish Speech runtime.

The application is designed for a reference recording supplied by the user and
local generation of script audio in English, Brazilian Portuguese, Spanish,
Hindi, Chinese, and French. Model weights, reference recordings, and generated
audio remain outside Git under the configured cache and data directories.

Setup and runtime commands will be added with the Fish Speech integration. The
first version is local-only and does not provide a hosted self-service API.
See `docs/superpowers/specs/2026-11-11-fish-speech-local-design.md` for the
approved scope and `docs/superpowers/plans/2026-11-11-fish-speech-local.md` for
the implementation plan.
