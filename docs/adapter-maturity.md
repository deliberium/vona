# Adapter Maturity

Vona separates the core runtime contract from backend and transport adapters. This document gives users a plain-language view of what each adapter is ready for.

| Surface | Status | Best For | Notes |
|---------|--------|----------|-------|
| `vona` | Facade crate | Application dependencies on crates.io | Re-exports `vona-core` and optional adapter crates through features. |
| `vona-core` | Core contract crate | Direct adapter development | Holds provider-neutral traits, session driver, runtime policy, skills, and types. |
| `RealtimeVoiceBackend` | Stable contract surface | Hosted realtime voice APIs, Moshi-family dialogue, open realtime voice adapters | Provider-neutral event-stream contract for full-duplex audio, tool calls, interruptions, and latency marks. |
| `PassthroughStsBackend` | Stable test utility | Contract tests, examples, fixture replay | Echoes input audio as output audio. Not a model backend. |
| `MockBackend`, `ScriptedTransport`, and `ScriptedRealtimeBackend` | Stable test utility | Deterministic runtime tests and examples | Lives in `vona-test-harness`; no external services required. |
| `vona-openai-realtime` | Experimental protocol adapter | OpenAI Realtime voice sessions | Maps Vona realtime input/output to provider JSON events and PCM16 audio chunks. Live WebSocket transport is intentionally outside deterministic tests. |
| `vona-gemini-live` | Experimental protocol adapter | Gemini Live native-audio sessions | Maps setup, realtime audio input, tool responses, and inline audio output. Uses the Live API server-to-server shape. |
| `vona-azure-speech` | Experimental protocol adapter | Azure Voice Live, Azure Speech STT/TTS | Includes Voice Live endpoint config plus Speech STT/TTS helper messages for cascaded backends. |
| `vona-elevenlabs` | Experimental speech component | Streaming TTS in cascaded ASR+LLM+TTS adapters | ElevenLabs WebSocket TTS is not treated as native STS by Vona. |
| `vona-deepgram` | Experimental speech component | Flux/listen STT and Aura streaming TTS in cascaded adapters | Deepgram is represented as speech components rather than a single native STS backend. |
| `vona-qwen` | Experimental protocol adapter | Qwen realtime voice integration work | Keeps Qwen-specific realtime protocol/config mapping outside `vona-core`. |
| `vona-ollama` | Experimental text component | Local loopback LLM generation in cascaded ASR+LLM+TTS adapters | Streams Ollama `/api/generate` chunks as Vona text frames. Requires a running Ollama server and installed model. |
| `vona-model-provisioning` | Experimental provisioning surface | Local model cache planning and artifact manifests | Lets Vona own local model layout decisions without forcing network downloads in core tests. |
| `vona-mlx` | Experimental local audio engine facade | Apple Silicon MLX audio experiments | Provides MLX-facing audio engine and streaming speech contracts. Native MLX features require macOS/Apple Silicon with Metal tooling. |
| `vona-mlx-speech` | Experimental loader utility crate | Shared native Rust MLX speech model loading | Keeps model parsing and tensor-loading utilities out of adapter facades. |
| `vona-mlx-whisper` | Experimental STT component | Native MLX Whisper/Distil-Whisper speech-to-text | Uses Vona model provisioning paths and explicit local artifacts; not yet a broad Whisper architecture matrix. |
| `vona-mlx-qwen3-tts` | Experimental TTS component | Native MLX Qwen3 text-to-speech | Uses native Rust loading and MLX execution; operator must provision compatible Qwen3 TTS artifacts. |
| `SeamlessM4tHttpBackend` | Experimental adapter | Process-isolated model serving through HTTP | Uses JSON and normalized `f32` PCM for bring-up simplicity. |
| `SeamlessM4tLocalBackend` | Experimental adapter | Embedded ONNX Runtime experiments | Requires operator-supplied ONNX artifacts and local ORT loading. |
| `vona-sidecar` HTTP API | Experimental deployment surface | Local sidecar experiments and integration tests | Supports optional bearer auth through `VONA_SIDECAR_AUTH_TOKEN`. |
| `vona-sidecar` Unix IPC API | Experimental deployment surface | Same-host sidecar experiments | Uses length-prefixed CBOR frames with size limits. |
| `vona-moshi` | Experimental adapter | Moshi-family WebSocket/Opus integration work | Requires Opus and a reachable Moshi-compatible service. Future Moshi-family work such as Unmute-style cascades should map onto `RealtimeVoiceBackend`. |
| Hosted realtime voice API adapters | Initial crates added | Production voice-agent services | OpenAI Realtime, Gemini Live, and Azure Voice Live have protocol/config crates; live transports should be gated behind integration tests with credentials. |
| Chroma/Covo-Audio-style open realtime adapters | Planned adapter family | Open realtime voice model runners | Represented by `RealtimeVoiceModelFamily::OpenRealtimeModel`; concrete adapters should land once serving contracts stabilize. |

## What Stable Means Here

Stable test utilities are intended to remain dependable for examples and integration tests. They do not imply a semver-stable public API before `1.0`.

Experimental adapters are useful for development and integration experiments, but callers should expect rough edges and API changes before production use.

## How To Add An Adapter

New adapters should:

- live in their own crate under `crates/`
- depend on `vona-core`, not on the `vona` facade or application-specific internals
- map provider failures into `BackendError` or transport-specific errors
- include deterministic tests that do not require external services
- document required environment variables and system dependencies
- keep local model downloads explicit through provisioning helpers; backend constructors should not surprise-download large artifacts
- keep authentication and product policy outside the core runtime contract

Realtime voice adapters should implement `RealtimeVoiceBackend` and test event ordering, interruption, tool-call, and close semantics.
