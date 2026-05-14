# Vona Roadmap

This roadmap is intentionally practical. Vona is useful today as a runtime contract, adapter workspace, deterministic harness, and crates.io facade for realtime voice infrastructure. The project should stay clear about what is ready for early adopters, what is experimental, and what would make production integrations easier.

## Current 0.1.0 Shape

The first open-source release is expected to include:

- `vona-core` as the provider-neutral runtime contract crate.
- `vona` as the umbrella facade crate with opt-in adapter features.
- `SpeechToSpeechBackend` for step-oriented speech-to-speech systems.
- `RealtimeVoiceBackend` for hosted realtime APIs, Moshi-family dialogue, and future open realtime voice models.
- Session metrics for first audio, tool calls, interruptions, fallback decisions, injected context events, and output-after-interruption checks.
- Skill registry, schema validation, audit events, and external context injection.
- Deterministic mock, scripted transport, realtime, benchmark, and release-gate harnesses.
- Seamless M4T-style HTTP, local ONNX, and sidecar surfaces.
- Moshi WebSocket/Opus backend surface.
- Local HTTP and Unix IPC transport helpers through `vona-transport-local` and `vona-sidecar`.
- Provider protocol/component crates for OpenAI Realtime, Gemini Live, Azure Voice Live/Speech, ElevenLabs TTS, and Deepgram STT/TTS.
- `vona-model-provisioning` with local cache planning, explicit HTTP provisioning, streamed temp-file downloads, size checks, SHA-256 verification, and atomic cache writes.
- Release automation for package checks, publish-order crates.io releases, changelog updates, version bumps, release gates, benchmark docs, and GitHub release creation.

These pieces are still pre-1.0. The contracts are ready for integration experiments and adapter development, but production deployments should treat provider/live-model paths as experimental until they are validated in the target environment.

## Near Term

- Publish the first tagged `v0.1.0` release through the GitHub release workflow after the release gate passes.
- Keep `vona-core` provider-neutral and small.
- Keep the `vona` facade crate focused on re-exports and optional features.
- Keep the model-free quick start and mock harness as the first contributor path.
- Expand deterministic examples around session lifecycle, interruptions, skill calls, context injection, fallback behavior, and realtime response completion.
- Add issue templates for bug reports, backend adapter proposals, provider integration notes, and documentation gaps.
- Add CI coverage that runs the release gate on Linux and macOS with Opus installed.
- Add credential-gated, ignored integration tests for live provider adapters once provider accounts, spend limits, and data-retention policies are explicit.
- Build cascaded backend examples from Deepgram/Azure STT, an application LLM, and ElevenLabs/Deepgram/Azure TTS while keeping the cascade behind Vona backend traits.
- Add provider-aware model provisioning helpers for Hugging Face and Ollama-style local model pulls.
- Add concrete adapters for newer open realtime voice models once their serving contracts stabilize.

## Backend And Transport Work

- Document adapter maturity per backend before each release.
- Keep HTTP sidecar support as the easiest deployment boundary for model-serving experiments.
- Keep local IPC support for same-host sidecars where JSON overhead is undesirable.
- Add a production transport adapter example once the host-application boundary is proven.
- Promote hosted realtime protocol crates toward live WebSocket adapters with explicit feature gates and integration-test guidance.
- Harden sidecar observability with structured health, readiness, and backend capability endpoints.
- Keep local ONNX and Moshi-family adapters aligned with `vona-model-provisioning` manifests so Vona owns cache layout and artifact validation.
- Avoid moving product policy into backend crates; host applications should own admission, wake, playback, auth, and UX decisions.

## Developer Experience

- Preserve a model-free quick start that runs without audio hardware or external services.
- Add more fixture-driven waveform tests for edge cases such as empty frames, channel mismatch, and interruption timing.
- Keep benchmark output deterministic enough for release comparison, while making clear that production SLOs must be measured in the target deployment.
- Prefer small examples that compile quickly over large demos that require model weights.
- Keep documentation aligned across `README.md`, adapter maturity, STS model coverage, cloud provider adapters, model provisioning, and release readiness docs.
- Add copy-pasteable examples for the facade crate features most applications are likely to start with.

## API Stability

The pre-1.0 API may still change. Changes should be motivated by one of these needs:

- a new backend cannot express required capabilities through existing traits
- runtime policy or audit behavior is ambiguous in real integrations
- transport boundaries need stronger safety or observability
- names or shapes are confusing for external contributors

Breaking changes should be documented in release notes once tagged releases begin.
