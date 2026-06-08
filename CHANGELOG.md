# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `vona-wake` crate with wake admission, wake-gated transport, pre-roll release, playback/privacy suppression, and optional speaker verification primitives.
- `vona/wake` facade feature that re-exports `vona-wake` for downstream applications.
- Live wake stream example and Criterion benchmark harness for validating wake gate behavior and hot-path performance.
- Template wake detector, explicit wake gate re-arm controls, and transport admission benchmark coverage for the Vona wake path.
- Generated-voice wake evaluation harness with persisted labeled WAV fixtures, clean/quiet/loud/noisy variants, reusable synthetic corpus manifest output, JSON report output, and optional enforcement mode.
- Manifest-driven real voice evaluation harness for externally recorded labeled WAV corpora, with human-recorded source provenance, calibration/evaluation split reporting, speaker-gated enrollment support, unauthorized-wake negative coverage, exact and gain-normalized corpus leakage checks, negative-category coverage, threshold sweep calibration, statistical confidence bounds, coverage and subgroup reporting/enforcement for speaker/environment/distance/device/session slices, and strict precision/recall, phrase-match, onset-relative latency, repeated-positive-wake-event, and false-wakes-per-hour gates with JSON report output.
- Real voice corpus builder script that validates or converts recorded audio fixtures, normalizes the corpus layout, requires per-row `source_type=human-recorded` by default, and emits a `real_voice_eval` manifest from CSV metadata with required corpus identity/version and optional collection-ledger metadata.
- Real voice corpus sizing planner for estimating positive-case and negative-audio requirements from the same statistical confidence assumptions used by the evaluator.
- Balanced real voice recording-plan generator that emits a corpus-builder CSV worklist and optional operator-facing Markdown guide for enrollment, early/mid/late positive wake onsets, unauthorized-wake rejection, and long negative/background recordings.
- Guided real voice recorder for walking a recording-plan CSV, rejecting malformed or non-human-source release rows before prompting, showing operators the required metadata, and writing 16 kHz mono WAVs to the planned paths through local recorder backends.
- Recording progress checker for planned real voice CSVs that rejects missing release-plan columns and reports missing, invalid, short, quiet, clipped, and non-human/missing source-provenance rows before corpus building, grouped by role, category, device, session, and source type.
- Real voice corpus readiness auditor for checking manifest size, collection-ledger consent/provenance plus device/session provenance coverage, enrollment/case WAV duration and audio quality, required phrase/template coverage, required category coverage and per-category exposure, per-speaker positive coverage including held-out enrollment-session positives, per-environment/distance/device/session exposure, wake onset bucket coverage, metadata coverage and subgroup balance, positive annotations, exact and gain-normalized duplicate/leakage risks, and confidence-derived corpus requirements before running the scorer.
- Real voice recording-plan auditor for catching invalid CSV columns, split/provenance gaps, weak wake phrase/template coverage, wake phrase label-semantic mistakes, missing or under-exposed release-grade categories, weak enrolled-speaker held-out-session and environment/distance/device coverage, missing wake onset buckets, speaker-gating mistakes, weak aggregate/evaluation-split metadata coverage, invalid positive annotations, and undersized evaluation exposure before audio collection starts.
- Real voice threshold selector that chooses an operating point from calibration split metrics, requires a stable passing calibration region by default, verifies the selected threshold against evaluation split metrics, and records the real-report hash for mandatory acceptance-time binding.
- Real voice acceptance checker for validating saved readiness, threshold-selection, and evaluation JSON artifacts, including independent audit provenance/leakage checks, real-report human-recorded provenance checks, saved metric/case-detail and confidence-bound consistency checks, manifest-content matching, calibration-selected threshold binding back to the real report's threshold sweep, and per-case evaluation subgroup gates for speakers, environments, distances, devices, sessions, categories, phrases, repeated positive wakes, and wake-onset buckets, before making a wakeword reliability claim.
- Unit coverage for real voice acceptance, collection-ledger, and evidence-status gates so weak evaluation subgroups, subgroup false-wake rates, missing consent/provenance coverage, and missing/accepted evidence states cannot regress silently.
- Real voice evidence status summarizer that reports passed/missing stages, release metrics, coverage, failures, and next actions from saved wake evaluation artifacts.
- Real voice evidence packager that bundles generated, readiness, acceptance, evaluation, and real-evidence-status reports with hashes, corpus identity, manifest SHA-256, stage status, coverage/provenance summaries, threshold-region counts, combined failures, next actions, and an acceptance summary for release review.
- Wake evaluation runner and evidence packager wired into the release gate, with generated regression checks always enabled, the full wake evidence Python suite, optional integrated real-corpus readiness auditing, calibration-based threshold-selection artifacts, real-corpus enforcement available through `VONA_WAKE_REAL_EVAL_MANIFEST`, and Markdown summary/status reporting for review artifacts, including nonzero failure plus status output when release jobs require a real manifest but none is supplied.
- Vona Wake architecture documentation with a Vona-themed SVG architecture diagram, downstream integration flow, and ownership boundaries for wake admission, speaker gating, and application policy.
- Qwen3 TTS streaming A/B harness that compares the full offline synthesis oracle against the selected streaming vocoder path with matching text/seed inputs, first-audio timing, total synthesis timing, waveform deltas, log-spectral deltas, JSON report output, and optional quality gates.
- Qwen3 TTS streaming vocoder mode selection through `VONA_MLX_QWEN3_TTS_STREAM_VOCODER_MODE`, with `prefix` for full-prefix oracle streaming, `rolling` for the current overlap-add fallback, and an experimental gated `cached` mode for incremental waveform decoder/cache experiments.
- Built-in Qwen3 TTS A/B text suite covering short, medium, long, punctuation-heavy, numeric, and conversational-fragment cases, with the rolling vocoder window default raised to 160 frames after smaller windows failed long or numeric suite cases.
- Provider-neutral realtime TTS policy and router for cached acknowledgement audio, Kokoro realtime replies by default, configurable downstream realtime TTS providers, Piper low-power/fallback synthesis, and Qwen3 premium synthesis, plus provider-managed provisioning manifests for the realtime TTS stack.
- `vona-kokoro-onnx` phonemizer-backed Kokoro ONNX realtime TTS adapter, `vona/kokoro-onnx` facade feature, Kokoro smoke example, and policy-wired MLX/Ollama voice benchmark coverage for the Kokoro default realtime path.
- Multi-model MLX/Ollama/MLX-VLM voice benchmark reporting through `VONA_E2E_OLLAMA_MODELS` and `VONA_E2E_MLX_VLM_MODELS`, including per-model timing, response, success, and load/error rows for configurable Vona backend-profile A/B runs.
- Persistent MLX-VLM text-generation worker support in `vona-mlx`, allowing Gemma 4 12B 4-bit MLX inference to stay loaded across Vona text-generation turns instead of spawning `mlx_vlm.generate` per request.
- Streaming MLX-VLM token deltas through Vona `TextGenerationFrame`s, first-frame latency reporting in the MLX/Ollama voice benchmark, configurable long-answer benchmark instructions, and cooperative cancellation when downstream consumers drop an in-flight generation stream.
- Provider-neutral text routing policy and `PolicyTextGenerator`, defaulting to Ollama `phi4-mini` for short low-latency voice turns and MLX-VLM Gemma 4 12B 4-bit for long, complex, or reasoning-weighted turns, with downstream backend overrides.
- Configurable native Whisper transcript hotwords through `TranscriptHotword`, `WhisperSpeechConfig::with_hotwords(...)`, and `VONA_WHISPER_HOTWORDS`, so downstream applications can normalize customer/project vocabulary without hard-coded Vona terms.
- Protected native Whisper worker runtime through `ProtectedWhisperTranscriber` and the `vona_mlx_whisper_worker` binary, isolating MLX C++/Metal aborts behind a restartable subprocess while keeping a model-loaded worker warm for repeated ASR turns.
- `vona-moonshine` native local STT binding and protected worker adapter plus `vona/moonshine` facade feature, using direct `libmoonshine` loading or a persistent Moonshine worker over raw `f32` PCM, configurable model arch/cache/hotwords, and benchmark-backed `MEDIUM_STREAMING` defaults.
- Manifest-driven 2,000-case generated-voice Moonshine ASR benchmark tooling, including deterministic corpus generation, native corpus scoring, category rollups, WER/RTF/p95 latency reporting, and documentation that separates generated regression evidence from human-recorded release evidence.
- MLX artifact prefetch script plus local `mlx-sys` Cargo patch support for `MLX_SOURCE_DIR`, allowing bootstrap/package builds to use a prefetched `ml-explore/mlx` checkout instead of CMake fetching from GitHub during native MLX configuration.

### Fixed

- Strengthened native Whisper transcript postprocessing for repeated ASR restarts that include concatenated phrase variants such as `Readycase`, `Checkcase`, or configured hotword variants, reducing repeated live-user turns without masking unrelated recognition errors.

## [0.2.0] - 2026-05-31

### Added

- `vona-core` generation contracts for text token streaming plus decoupled audio transcription and synthesis traits.
- `vona-ollama` loopback text adapter for Ollama `/api/generate` streaming, exposed through the umbrella `vona/ollama` feature.
- `vona-mlx` Apple Silicon MLX audio engine facade, optional `mlx-models` text-loader hook, and umbrella MLX feature exports.
- `vona-mlx-speech` shared native Rust speech model loading utilities for safetensors discovery, metadata parsing, and MLX tensor materialization.
- `vona-mlx-whisper` native MLX Whisper loader and smoke example surface for local speech recognition experiments.
- `vona-mlx-qwen3-tts` native MLX Qwen3 TTS loader and smoke example surface for local speech synthesis.
- Vona model-provisioning manifests for Distil-Whisper Large V3 and the bf16 MLX Qwen3 TTS checkpoint so downstream apps can explicitly download speech assets before loading `vona-mlx`.
- `mlx_ollama_voice_bench` example and MLX/Ollama benchmark documentation covering the local voice pipeline validation flow.

### Changed

- Split the MLX backend surface into granular contracts and crates so Ollama text generation, MLX audio orchestration, shared speech loading, Whisper, and Qwen3 TTS remain independently optional.
- Split `vona-mlx` feature gates so `native-mlx` enables only the MLX runtime and `mlx-models-loader` opts into the heavier `mlx-models` and tokenizer text-loader stack.
- Narrowed workspace `reqwest` defaults so local Ollama-only builds do not inherit TLS support, while remote HTTPS crates opt into Rustls explicitly.
- Removed unused direct dependency surface from `vona-mlx` after the bloat review without changing runtime behavior.

### Fixed

- Corrected native Qwen3 TTS vocoder transposed-convolution stride inference so 12.5 Hz speech tokens decode to the expected 24 kHz waveform duration.
- Corrected the Ollama Phi mini default tag to `phi4-mini`, matching the Ollama library tag accepted by `ollama pull`.
- Updated benchmark notes to mark the earlier long-run Qwen3 TTS timings as historical because they were collected before the stride fix.

## [0.1.1] - 2026-05-16

### Added

- `vona-qwen` crate with Qwen realtime ASR/TTS configuration helpers and event mapping utilities.

## [0.1.0] - 2026-05-14

### Added

- Initial Vona speech-to-speech runtime contracts, session driver, runtime policy, skill registry, passthrough backend, Seamless adapters, Moshi adapter surface, local transport helpers, sidecar, and deterministic test harness.
- Cloud provider protocol crates for OpenAI Realtime, Gemini Live, Azure Voice Live/Speech, ElevenLabs, and Deepgram.
- `vona-model-provisioning` for local model manifests, cache planning, and explicit HTTP artifact provisioning.
- Umbrella `vona` features for cloud adapters and local model provisioning.
- Release automation for version bumping, changelog maintenance, package checks, and publish-order crates.io releases.
- Integrity checks for model provisioning artifacts using expected size and SHA-256 metadata.

### Changed

- Split the facade crate so `vona-core` owns provider-neutral runtime contracts while `vona` re-exports stable public surfaces and optional adapters.
- Expanded the release gate to cover provider crates and model provisioning.
- Remote Seamless sidecar responses now preserve all output audio frames while retaining the original single-frame fields for compatibility.

### Fixed

- CI clippy compatibility for newer Rust toolchains in the Moshi WebSocket receive loop.
- Hosted realtime response completion is no longer treated as session closure.

[Unreleased]: https://github.com/deliberium/vona/compare/v0.2.0...HEAD
[0.1.0]: https://github.com/deliberium/vona/releases/tag/v0.1.0
[0.1.1]: https://github.com/deliberium/vona/releases/tag/v0.1.1
[0.2.0]: https://github.com/deliberium/vona/releases/tag/v0.2.0
