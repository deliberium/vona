# Vona

Vona is a Rust workspace for building real-time speech-to-speech runtimes.

It provides the contracts, session loop, transport plumbing, backend adapters, and deterministic test harness needed to embed low-latency voice systems without tying application code to one model provider or serving topology.

Vona is intentionally not a complete voice assistant. Host applications remain responsible for admission control, wake policy, playback policy, authentication, product UX, and deployment. Vona owns the reusable runtime boundary between audio transports, speech-to-speech backends, skill/tool context, and fallback policy.

## Why Vona

Most speech-to-speech projects begin as one of three things: a model demo, a provider SDK wrapper, or application-specific voice-agent glue. Vona is aimed at a different layer.

Use Vona when you want:

- a Rust-native boundary between audio transports and speech-to-speech backends
- deterministic tests for interruption, tool-call, context-injection, and fallback behavior
- the option to run model backends in-process, behind HTTP, or behind local IPC
- provider-neutral traits that let one host application try multiple STS backends
- a small core crate that does not own your product policy or UX

Do not use Vona if you need a turnkey assistant, hosted model service, wake-word engine, audio device stack, or production WebRTC integration out of the box.

## What Is In This Repository

| Crate | Purpose |
|-------|---------|
| `vona` | Core traits, event types, session driver, runtime policy, skill registry, and passthrough backend. |
| `vona-seamless` | Seamless M4T-style local ONNX and HTTP sidecar backend adapters. |
| `vona-moshi` | Kyutai Moshi backend surface using WebSocket and Opus framing. |
| `vona-transport-local` | Local HTTP/IPC transport helpers and length-prefixed CBOR framing. |
| `vona-sidecar` | Sidecar binary exposing Vona backends over HTTP and Unix-socket IPC. |
| `vona-test-harness` | Deterministic mock backend, scripted transport, fixtures, and benchmark harnesses. |

The workspace is backend-agnostic by design. Provider-specific integrations live in adapter crates; the core `vona` crate stays focused on stable contracts.

## Current Status

Vona is pre-1.0 and suitable for integration experiments, adapter development, and deterministic runtime testing. The public APIs may still change before a stable release.

Implemented today:

- step-oriented speech-to-speech backend trait
- audio transport trait
- session driver with metrics for first audio, tool calls, interruptions, and fallback decisions
- skill execution registry with schema validation and audit events
- context injection through `ExternalContextEvent`
- passthrough, Seamless M4T-style, Moshi, HTTP sidecar, and local IPC surfaces
- deterministic test harnesses and release-gate benchmarks

Known limits:

- production transport adapters such as LiveKit are not included yet
- the Seamless local ONNX path depends on operator-supplied model artifacts
- text-conditioned local generation is not yet parity-complete with all deployment modes
- performance SLOs beyond the deterministic release gate should be measured in your target environment

## Prerequisites

Vona is a Rust workspace. Install a recent Rust toolchain with Cargo.

`vona-moshi` links against Opus:

```bash
# macOS
brew install opus

# Debian/Ubuntu
sudo apt-get install libopus-dev pkg-config
```

If Opus is installed in a non-standard prefix, set `LIBOPUS_LIB_DIR` to the prefix path, not the raw `lib` directory:

```bash
export LIBOPUS_LIB_DIR=/opt/homebrew
```

## Quick Start

Clone the repository and run the deterministic release gate:

```bash
git clone https://github.com/deliberium/vona.git
cd vona
bash scripts/release_gate.sh
```

For a faster inner loop while developing:

```bash
cargo check --workspace --all-targets --locked
cargo test -p vona --locked
cargo test -p vona-test-harness --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Run the deterministic mock harness:

```bash
cargo test -p vona-test-harness waveform_fixture_round_trips_through_scripted_transport -- --nocapture
```

## Minimal Backend Example

The core backend contract is step-oriented. A backend receives an `AudioInputFrame`, returns zero or more `AudioOutputFrame`s, and may emit control events for the runtime to handle.

```rust
use async_trait::async_trait;
use vona::{
    AudioInputFrame, AudioOutputFrame, BackendCapabilities, BackendError, BackendStep,
    ExternalContextEvent, SessionConfig, SpeechToSpeechBackend,
};

#[derive(Debug, Clone, Default)]
struct MyBackend;

#[async_trait]
impl SpeechToSpeechBackend for MyBackend {
    type Session = SessionConfig;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(config)
    }

    async fn step(
        &self,
        _session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        Ok(BackendStep {
            output_audio: vec![AudioOutputFrame {
                sequence: input.sequence,
                sample_rate_hz: input.sample_rate_hz,
                channels: input.channels,
                samples: input.samples,
                is_filler: false,
            }],
            ..BackendStep::default()
        })
    }

    async fn inject_event(
        &self,
        _session: &mut Self::Session,
        _event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    async fn end_session(&self, _session: Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}
```

For a ready-made deterministic implementation, use `PassthroughStsBackend` from the `vona` crate or `MockBackend` from `vona-test-harness`.

## Runtime Model

The runtime loop connects four surfaces:

- `AudioTransport`: receives input frames, sends output frames, and clears buffered output on interruption
- `SpeechToSpeechBackend`: owns provider/model session state and performs each audio step
- `VonaRuntime`: applies policy to backend control events
- `SkillExecutor`: resolves tool calls and injects external context back into the backend

The important integration primitive is `ExternalContextEvent`. It carries transcript overrides, tool results, planner output, precomputed replies, or other application-owned context without forcing the core backend trait to know about any one product.

See [docs/architecture.md](docs/architecture.md) for the sidecar contract and request/response shapes.

## Sidecar And Local Backends

The `vona-sidecar` binary exposes the Seamless M4T-style backend over HTTP and, on Unix platforms, a local IPC socket.

Default HTTP bind:

```bash
VONA_STS_SIDECAR_BIND=127.0.0.1:9090
```

Health check:

```bash
curl --silent --fail http://127.0.0.1:9090/healthz
```

Local Seamless M4T ONNX configuration:

```bash
export VONA_STS_ONNX_MODEL_PATH=/absolute/path/to/seamless_m4t.onnx
export VONA_STS_ONNX_INPUT_NAME=audio
export VONA_STS_ONNX_OUTPUT_NAME=waveform
export VONA_STS_ONNX_SAMPLE_RATE=16000
```

See [docs/production-backends.md](docs/production-backends.md) for operational expectations and current limitations.

Adapter maturity is tracked in [docs/adapter-maturity.md](docs/adapter-maturity.md).

## Model-Free Demo

You can run a complete Vona session without model weights, network access, or audio hardware:

```bash
cargo run -p vona-test-harness --example mock_session --locked
```

The demo drives a scripted audio frame through the runtime, emits a mock skill call, handles an interruption, injects tool context back into the backend, and prints the resulting session metrics.

## Release Gate

The release gate is the source of truth for pre-release validation:

```bash
bash scripts/release_gate.sh
```

It runs:

- locked workspace checks
- deterministic per-crate tests
- all-target compile checks
- clippy with `-D warnings`
- deterministic transport smoke benchmarks
- benchmark result generation in [docs/benchmark-results.md](docs/benchmark-results.md)

Read the full checklist in [docs/release-readiness-checklist.md](docs/release-readiness-checklist.md).

## Repository Layout

```text
crates/
  vona/                  core runtime contracts
  vona-seamless/         Seamless M4T-style backend adapters
  vona-moshi/            Moshi backend surface
  vona-transport-local/  local IPC and transport helpers
  vona-sidecar/          sidecar binary
  vona-test-harness/     deterministic tests and benchmarks
docs/                    architecture, backend, benchmark, and release docs
examples/                example slots and fixture-driven demos
tests/fixtures/          deterministic waveform fixtures
scripts/                 release and maintenance scripts
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

Useful rules of thumb:

- keep core contracts provider-neutral
- put provider integrations in adapter crates
- include deterministic tests for runtime, transport, or backend behavior
- keep `bash scripts/release_gate.sh` green

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

The current roadmap is in [docs/roadmap.md](docs/roadmap.md).

## Security

Please do not open public issues for security vulnerabilities. Report them using the process in [SECURITY.md](SECURITY.md).

## License

Vona is licensed under the [MIT License](LICENSE).
