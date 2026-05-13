# Adapter Maturity

Vona separates the core runtime contract from backend and transport adapters. This document gives users a plain-language view of what each adapter is ready for.

| Surface | Status | Best For | Notes |
|---------|--------|----------|-------|
| `PassthroughStsBackend` | Stable test utility | Contract tests, examples, fixture replay | Echoes input audio as output audio. Not a model backend. |
| `MockBackend` and `ScriptedTransport` | Stable test utility | Deterministic runtime tests and examples | Lives in `vona-test-harness`; no external services required. |
| `SeamlessM4tHttpBackend` | Experimental adapter | Process-isolated model serving through HTTP | Uses JSON and normalized `f32` PCM for bring-up simplicity. |
| `SeamlessM4tLocalBackend` | Experimental adapter | Embedded ONNX Runtime experiments | Requires operator-supplied ONNX artifacts and local ORT loading. |
| `vona-sidecar` HTTP API | Experimental deployment surface | Local sidecar experiments and integration tests | Supports optional bearer auth through `VONA_SIDECAR_AUTH_TOKEN`. |
| `vona-sidecar` Unix IPC API | Experimental deployment surface | Same-host sidecar experiments | Uses length-prefixed CBOR frames with size limits. |
| `vona-moshi` | Experimental adapter | Moshi WebSocket/Opus integration work | Requires Opus and a reachable Moshi-compatible service. |

## What Stable Means Here

Stable test utilities are intended to remain dependable for examples and integration tests. They do not imply a semver-stable public API before `1.0`.

Experimental adapters are useful for development and integration experiments, but callers should expect rough edges and API changes before production use.

## How To Add An Adapter

New adapters should:

- live in their own crate under `crates/`
- depend on `vona`, not on application-specific internals
- map provider failures into `BackendError` or transport-specific errors
- include deterministic tests that do not require external services
- document required environment variables and system dependencies
- keep authentication and product policy outside the core runtime contract
