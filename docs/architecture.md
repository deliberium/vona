# Vona-rs Architecture

The workspace is split into two layers:

- Core runtime contracts in `crates/vona-core`
- Facade re-exports and optional adapter features in `crates/vona`
- Deterministic verification tooling in `crates/vona-test-harness`

## Layer Boundaries

`crates/vona-core` owns the generic speech-to-speech execution model:

- session configuration and runtime policy
- transport and backend traits
- event-stream realtime voice traits
- skill injection and external context event plumbing
- backend adapters such as passthrough and HTTP sidecars

`crates/vona` is the umbrella crate for crates.io consumers. It re-exports `vona-core` by default and exposes adapter crates through opt-in features.

`crates/vona-test-harness` exists to verify backend and runtime behavior without a live model server. It provides deterministic transports, scripted backends, and mock skill execution so runtime policy can be exercised in tests.

## Runtime Contracts

Vona intentionally has two backend shapes:

- `SpeechToSpeechBackend` for step-oriented systems such as Seamless-style speech translation.
- `RealtimeVoiceBackend` for event-stream systems such as hosted realtime voice APIs, Moshi-family full-duplex dialogue, and new open realtime voice models.

The realtime contract uses explicit input and output events for audio chunks, text, tool results, control messages, tool calls, interruptions, latency marks, and close events. This keeps provider-specific WebSocket, IPC, or hosted-service event formats out of host application code.

## Sidecar Contract

The first real provider path is an HTTP sidecar for Seamless M4T-style turns. The adapter currently expects a step-oriented request/response loop with caller-supplied session identifiers and sideband context events.

Base URL:

- configured through `VONA_STS_BASE_URL`
- `SeamlessM4tHttpBackend` posts to `/v1/seamless-m4t/step`

Request shape:

```json
{
  "session_id": "voice-session-123",
  "sample_rate_hz": 16000,
  "channels": 1,
  "input_samples": [0.12, -0.04, 0.3],
  "model": "facebook/hf-seamless-m4t-medium",
  "session_metadata": {
    "user_id": "ava",
    "thread_id": "voice-thread-1",
    "provenance": "assistant_active_conversation"
  },
  "style_profile": {
    "pace": 1.02,
    "warmth": 0.58,
    "expressiveness": 0.61,
    "formality": "balanced",
    "preferred_voice": "speaker:0"
  },
  "pending_events": [
    {
      "source": "vona.plan_result",
      "spoken_summary": "The lights are on.",
      "payload": {
        "answer": "The lights are on.",
        "confidence": 0.91
      }
    }
  ]
}
```

Request fields:

- `session_id`: stable logical turn/session identifier chosen by the caller
- `sample_rate_hz`: sample rate of the current audio frame buffer
- `channels`: channel count of the current audio frame buffer
- `input_samples`: normalized `f32` PCM samples for the current step window
- `model`: optional provider-specific model identifier
- `session_metadata`: caller-supplied metadata carried from the owning application
- `style_profile`: optional normalized speech style hints
- `pending_events`: serialized external context events that the sidecar may use for grounding, prompt shaping, precomputed reply text, or dialogue state

Response shape:

```json
{
  "output_samples": [0.03, 0.09, -0.02],
  "output_sample_rate_hz": 16000,
  "transcript": "turn on the office lights",
  "control_events": [],
  "finished": false,
  "debug_payload": {
    "backend_mode": "t2st",
    "reply_text": "The lights are on.",
    "model_id": "facebook/hf-seamless-m4t-medium"
  }
}
```

Response fields:

- `output_samples`: normalized `f32` PCM samples generated for immediate playback
- `output_sample_rate_hz`: playback sample rate of the returned audio buffer
- `transcript`: optional normalized text for observability or downstream policy
- `control_events`: backend-originated control signals consumed by the runtime
- `finished`: whether the backend considers the active turn complete
- `debug_payload`: optional provider-specific diagnostics or reply metadata

## Sidecar Expectations

- The sidecar must treat `session_id` as the unit of backend state.
- The sidecar should be idempotent for retried requests where possible.
- The sidecar should accept context-only continuation steps where `input_samples` is empty but pending events carry a precomputed reply or transcript override.
- The sidecar should return transport-level failures as non-2xx responses and model/runtime failures as structured error bodies when available.
- The adapter currently assumes normalized `f32` PCM samples and JSON transport for bring-up simplicity, not final throughput optimization.
- Authentication is intentionally left deployment-specific. Host applications can inject auth through sidecar-aware reverse proxies or future adapter configuration.
