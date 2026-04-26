# Vona Production Backends

Status: current as of 2026-04-04

This document describes the currently implemented production-facing Vona backends and the operational expectations around them.

## Implemented Backends

Current backend surfaces in the workspace:

- `PassthroughStsBackend`: deterministic or trivial adapter used for tests and bring-up.
- `SeamlessM4tHttpBackend`: HTTP adapter that posts turn steps to `/v1/seamless-m4t/step`.
- `SeamlessM4tLocalBackend`: model-backed local backend that runs an in-process ONNX Runtime path via the ort crate.

## Session Model

The backend contract is step-oriented rather than transport-owned.

Every backend session is driven by:

- `SessionConfig`
- one or more `AudioInputFrame` steps
- optional `ExternalContextEvent` injections between steps
- a final `end_session()` call

This is the key operational boundary:

- Vona owns the generic STS runtime contract
- the embedding application owns admission, wake policy, playback policy, trust, and session lifecycle

## HTTP Sidecar Expectations

`SeamlessM4tHttpBackend` currently expects:

- JSON over HTTP
- normalized `f32` PCM samples
- explicit `sample_rate_hz` and `channels`
- caller-supplied `session_metadata`
- optional `style_profile`
- optional pending context events for transcript overrides or precomputed reply text

The current sidecar binary in the workspace is `vona-sidecar`.

Default local bind:

```bash
VONA_STS_SIDECAR_BIND=127.0.0.1:9090
```

Health check:

```bash
curl --silent --fail http://127.0.0.1:9090/healthz
```

## Local Seamless Backend Expectations

`SeamlessM4tLocalBackend` now depends on ONNX Runtime model artifacts and local ORT loading.

Default model metadata remains:

```bash
VONA_STS_MODEL=facebook/hf-seamless-m4t-medium
```

Required ONNX runtime configuration:

```bash
VONA_STS_ONNX_MODEL_PATH=/absolute/path/to/seamless_m4t.onnx
VONA_STS_ONNX_INPUT_NAME=audio
VONA_STS_ONNX_OUTPUT_NAME=waveform
VONA_STS_ONNX_SAMPLE_RATE=16000
```

The current implementation supports two effective execution modes inside the same backend:

- audio-driven STS when raw audio is the primary input
- text-conditioned speech generation when the caller injects a precomputed reply or transcript override through context events

That second mode allows host applications to preserve higher-level orchestration while replacing a legacy ASR/TTS loop with a real STS-capable model backend.

## Failure Semantics

Backends should map failures into the Vona error model rather than panicking:

- `BackendError::Start` for model/session startup failure
- `BackendError::Step` for inference or transport step failure
- `BackendError::Inject` for context injection failure
- `BackendError::End` for session teardown failure

For HTTP backends specifically:

- non-2xx responses are treated as backend step failures
- malformed JSON responses are treated as backend step failures
- deployment-specific auth should be provided by surrounding infrastructure or future adapter configuration

## Current Limits

These production-facing limits still apply:

- no real transport adapter crate has landed yet for LiveKit or another remote session transport
- text-conditioned local generation is not yet parity-complete in the ONNX path and still expects audio-driven steps
- full latency, interruption, and fallback benchmark suites are still incomplete
- the sidecar contract is intentionally JSON-first for bring-up, not a final throughput-optimized wire format

## Recommended Usage Today

For applications embedding Vona today:

- use the HTTP sidecar when you want process isolation and a clear deployment seam
- use the local backend when you need an embedded bring-up path without the HTTP hop
- keep application-level fallback logic outside the backend itself
- treat `ExternalContextEvent` as the stable integration seam for transcript overrides, tool results, and precomputed reply summaries
