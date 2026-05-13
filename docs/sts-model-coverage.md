# STS Model Coverage

Vona supports more than one kind of speech-to-speech system. This distinction matters because the runtime contract for a translation model is not the same as the runtime contract for a full-duplex conversational voice model or a hosted realtime voice-agent API.

## Model Families

| Family | Vona Surface | Current Coverage | Notes |
|--------|--------------|------------------|-------|
| Multilingual translation STS | `SpeechToSpeechBackend`, `vona-seamless` | Experimental adapter | Fits SeamlessM4T / SeamlessM4T v2 style step-oriented translation and speech generation flows. |
| Full-duplex spoken dialogue | `RealtimeVoiceBackend`, `vona-moshi` | Experimental adapter plus provider-neutral realtime trait | Fits Moshi-family models where listening, speaking, interruption, and tool events can overlap. |
| Hosted realtime voice APIs | `RealtimeVoiceBackend`, `vona-openai-realtime`, `vona-gemini-live`, `vona-azure-speech` | Initial provider protocol crates | Fits APIs that stream audio directly and expose tool calls, barge-in, and response-control events. |
| Open realtime voice models | `RealtimeVoiceBackend` | Placeholder model family, no model-specific adapter yet | Intended for newer open realtime voice models such as Chroma/Covo-Audio-style systems when a stable Rust integration boundary is chosen. |
| Cascaded ASR + LLM + TTS | Adapter-specific, `vona-elevenlabs`, `vona-deepgram`, `vona-azure-speech` | Initial speech-component crates | Can be implemented as an adapter, but Vona should keep the cascade behind the same backend/runtime boundary. |

## Two Runtime Contracts

### Step-Oriented STS

Use `SpeechToSpeechBackend` when the backend is naturally driven by request/response turns:

- one audio frame or window goes in
- zero or more output frames come back
- control events are returned with the step
- context can be injected between steps

This is the right shape for Seamless-style translation backends and simple sidecars.

### Realtime Voice

Use `RealtimeVoiceBackend` when the backend is event-stream oriented:

- audio input and audio output may overlap
- tool calls can be emitted while audio continues
- interruptions and output clearing are explicit events
- latency marks are part of the session stream
- hosted services and local full-duplex models can share the same host-facing shape

This is the right shape for Moshi-family dialogue systems, hosted realtime voice APIs, and new open realtime voice models.

## Current Priority

Vona should remain honest about coverage:

- Seamless is the multilingual translation anchor.
- Moshi is the open full-duplex dialogue anchor.
- Hosted realtime APIs are represented by provider-specific protocol crates for OpenAI Realtime, Gemini Live, and Azure Voice Live.
- ElevenLabs and Deepgram are represented as speech-component crates for cascaded backends rather than native STS backends.
- Open realtime voice models are represented as a model-family target, but need concrete adapter crates once their Rust serving stories stabilize.
- Local model adapters should use `vona-model-provisioning` manifests so Vona owns cache layout, artifact validation, and future download policy.

## Adapter Expectations

Realtime adapters should map provider-native events into:

- `RealtimeVoiceInput::Audio`
- `RealtimeVoiceInput::ToolResult`
- `RealtimeVoiceInput::Control`
- `RealtimeVoiceOutput::Audio`
- `RealtimeVoiceOutput::ToolCall`
- `RealtimeVoiceOutput::Interruption`
- `RealtimeVoiceOutput::LatencyMark`
- `RealtimeVoiceOutput::Closed`

Adapters should include deterministic tests for event ordering, barge-in, tool-call propagation, context injection, and close semantics without requiring live services.
