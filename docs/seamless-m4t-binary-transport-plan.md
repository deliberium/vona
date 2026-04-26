# Seamless M4T Binary Transport Plan

## Goal

Replace the current local sidecar hop of:

- JSON request/response bodies
- `Vec<f32>` audio payload expansion
- loopback HTTP framing

with a faster local transport, while keeping core `vona` transport-agnostic and reusable across:

- desktop apps
- mobile companions
- server-side orchestrators
- non-application-specific integrations

## Design Principle

`vona` core should define the backend RPC contract, not the wire format.

That means:

- core owns the request/response schema
- core owns the backend-facing transport trait
- specific wire transports live behind implementations of that trait
- local binary transport is optional, not mandatory

## What Was Landed

Core `vona` now exposes a reusable Seamless remote transport boundary:

- `SeamlessM4tRemoteTransport`
- `SeamlessM4tRemoteStepRequest`
- `SeamlessM4tRemoteStepResponse`
- `SeamlessM4tRemoteBackend`

HTTP is now just one transport implementation layered on top of that abstraction.

## Recommended Next Step

Add a separate transport implementation crate, for example:

- `crates/vona-transport-local`

That crate should provide:

- Unix domain socket transport on Unix-like systems
- named pipe transport on Windows
- optional loopback TCP fallback where local IPC is unavailable

## Binary Protocol Recommendation

Use length-prefixed binary messages over a stream transport.

Recommended framing:

1. `u32` little-endian payload length
2. payload bytes

Recommended payload encoding:

- `postcard` if we want smallest payloads and can accept tighter schema coupling
- `bincode` if we want simpler Rust-first implementation
- `protobuf` if we want stronger cross-language interoperability

For the current shape of the integration, `protobuf` is the best long-term default if we expect non-Rust clients. `bincode` is the fastest short-term path if the first binary transport remains Rust-to-Rust.

## Audio Representation Recommendation

Do not send local sidecar audio as JSON floats.

Preferred local representations:

- `pcm_s16le` bytes for raw microphone and playback chunks
- optional `f32` only inside model-facing in-process code

Suggested request contract evolution:

- add `audio_encoding`
- add `audio_bytes`
- keep float samples only as an optional compatibility path

This keeps the public backend contract flexible enough for:

- raw PCM
- compressed audio
- future streaming chunk reuse

## Compatibility Strategy

Do not break the current HTTP endpoint immediately.

Phase the rollout:

1. Keep HTTP+JSON transport working as the compatibility baseline.
2. Add binary local transport behind a new implementation of `SeamlessM4tRemoteTransport`.
3. Let host applications choose the transport at construction time.
4. Optionally add feature-gated transport crates so lean builds do not pull in every dependency.

## Suggested Crate Boundaries

`crates/vona`

- backend traits
- request/response contract structs
- generic remote backend
- convenience HTTP transport wrapper

`crates/vona-sidecar`

- model host
- HTTP server adapter
- future local binary server adapter

`crates/vona-transport-local`

- client transport for UDS / named pipes / local TCP
- codec and framing logic

## Host Application Migration Path

Host applications should stop thinking in terms of “HTTP backend vs local backend”.

Instead:

- choose backend model implementation separately
- choose transport separately

Example matrix:

- local model + in-process backend
- local model + local binary sidecar transport
- local model + HTTP sidecar transport
- remote hosted transport

## Concrete Patch Sequence

1. Define protobuf or binary codec structs mirroring `SeamlessM4tRemoteStepRequest/Response`.
2. Add a binary server adapter to `vona-sidecar`.
3. Add a `SeamlessM4tLocalIpcTransport` client implementation in a separate crate.
4. Add host application config selection for transport kind:
   - `http`
   - `local_ipc`
5. Keep the existing HTTP path as fallback and debugging path.
6. Add transport benchmarks measuring:
   - payload size
   - encode/decode time
   - end-to-end step latency
   - CPU utilization

## Why This Stays Reusable

This approach avoids coupling `vona` to:

- Axum
- HTTP specifically
- Unix sockets specifically
- application-specific runtime choices

Other integrations can provide their own transport implementation as long as they satisfy the same remote step contract.
