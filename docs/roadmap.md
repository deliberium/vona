# Vona Roadmap

This roadmap is intentionally practical. Vona is useful today as a runtime contract and adapter workspace, but the project should be clear about what is stable, what is experimental, and what would make it easier for new users to adopt.

## Near Term

- Publish a first tagged `v0.1.0` release with the release gate passing on macOS and Linux.
- Keep `vona-core` provider-neutral and small.
- Keep the `vona` facade crate focused on re-exports and optional features.
- Expand deterministic examples around session lifecycle, interruptions, skill calls, context injection, and fallback behavior.
- Promote the initial OpenAI Realtime, Gemini Live, and Azure Voice Live protocol crates from deterministic protocol mapping to live transport adapters.
- Build cascaded backend examples from Deepgram/Azure STT, an application LLM, and ElevenLabs/Deepgram/Azure TTS.
- Extend `vona-model-provisioning` from cache planning to checksum-verified downloads and local provider pulls.
- Add concrete adapters for newer open realtime voice models once their serving contracts stabilize.
- Add CI jobs that run the release gate on Linux and macOS with Opus installed.
- Add issue templates for bug reports, backend adapter proposals, and documentation gaps.

## Backend And Transport Work

- Document adapter maturity per backend before each release.
- Add a production transport adapter example once the host-application boundary is proven.
- Keep HTTP sidecar support as the easiest deployment boundary for model-serving experiments.
- Keep local IPC support for same-host sidecars where JSON overhead is undesirable.
- Avoid moving product policy into backend crates; host applications should own admission, wake, playback, auth, and UX decisions.

## Developer Experience

- Preserve a model-free quick start that runs without audio hardware or external services.
- Add more fixture-driven waveform tests for edge cases such as empty frames, channel mismatch, and interruption timing.
- Keep benchmark output deterministic enough for release comparison, while making clear that production SLOs must be measured in the target deployment.
- Prefer small examples that compile quickly over large demos that require model weights.

## API Stability

The pre-1.0 API may still change. Changes should be motivated by one of these needs:

- a new backend cannot express required capabilities through existing traits
- runtime policy or audit behavior is ambiguous in real integrations
- transport boundaries need stronger safety or observability
- names or shapes are confusing for external contributors

Breaking changes should be documented in release notes once tagged releases begin.
