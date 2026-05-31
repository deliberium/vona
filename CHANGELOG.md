# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/deliberium/vona/compare/v0.1.1...HEAD
[0.1.0]: https://github.com/deliberium/vona/releases/tag/v0.1.0

[0.1.1]: https://github.com/deliberium/vona/releases/tag/v0.1.1
