# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/deliberium/vona/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deliberium/vona/releases/tag/v0.1.0
