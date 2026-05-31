<p align="center">
  <img src="resources/VonaLogo.png" alt="Vona logo" width="220">
</p>

<p align="center">
  <a href="https://github.com/deliberium/vona/actions/workflows/ci.yml"><img src="https://github.com/deliberium/vona/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

# Vona

Vona is the Rust runtime layer for the next wave of voice-native products: fast, composable, provider-neutral speech-to-speech infrastructure you can actually ship.

It gives teams the durable core that most voice prototypes end up rebuilding by hand: realtime session orchestration, audio transport boundaries, backend adapters, tool/context hooks, fallback policy, and deterministic harnesses for the moments that matter most, like interruption, first audio latency, tool calls, and event ordering.

Bring your own product surface, model strategy, deployment topology, and user experience. Vona owns the hard runtime boundary between microphones, transports, speech-to-speech models, local/cloud providers, skills, and policy so your application can move across backends without rewriting its voice stack.

## Why Vona

Most speech-to-speech projects start as a model demo, a provider SDK wrapper, or a tangle of application-specific voice-agent glue. That works until you need to swap models, run locally, move to a hosted realtime API, test interruptions, or prove latency before the launch window closes.

Vona is built for that inflection point. It is not another assistant template; it is the runtime substrate underneath one. The goal is simple: make voice systems feel as modular, testable, and backend-portable as the rest of a modern AI stack.

Use Vona when you want:

- a Rust-native boundary between audio transports and speech-to-speech backends
- first-class contracts for both step-oriented STS and event-stream realtime voice
- deterministic tests for interruption, tool-call, context-injection, and fallback behavior
- the option to run model backends in-process, behind HTTP, or behind local IPC
- provider-neutral traits that let one host application try multiple STS backends
- a small core crate that does not own your product policy or UX

Do not use Vona if you need a turnkey assistant, hosted model service, wake-word engine, audio device stack, or production WebRTC integration out of the box.

## What Is In This Repository

| Crate | Purpose |
|-------|---------|
| `vona` | Umbrella crate that re-exports `vona-core` and optional adapter crates through features. |
| `vona-core` | Core traits, event types, session driver, runtime policy, skill registry, and passthrough backend. |
| `vona-openai-realtime` | OpenAI Realtime protocol mapping for Vona realtime sessions. |
| `vona-gemini-live` | Gemini Live protocol mapping for Vona realtime sessions. |
| `vona-azure-speech` | Azure Voice Live plus Azure Speech STT/TTS helper surfaces. |
| `vona-elevenlabs` | ElevenLabs streaming text-to-speech helper surface for cascaded voice backends. |
| `vona-deepgram` | Deepgram Flux/listen STT and Aura streaming TTS helper surfaces. |
| `vona-qwen` | Qwen realtime voice protocol helper surface. |
| `vona-ollama` | Local Ollama loopback text-generation adapter for cascaded ASR+LLM+TTS systems. |
| `vona-model-provisioning` | Local model manifest and cache planning for Vona-owned model provisioning. |
| `vona-mlx` | Apple Silicon MLX audio engine facade and streaming STT/TTS contracts. |
| `vona-mlx-speech` | Shared native Rust MLX speech model loading utilities. |
| `vona-mlx-whisper` | Native Rust MLX Whisper speech-to-text loader and inference surface. |
| `vona-mlx-qwen3-tts` | Native Rust MLX Qwen3 text-to-speech loader and inference surface. |
| `vona-seamless` | Seamless M4T-style local ONNX and HTTP sidecar backend adapters. |
| `vona-moshi` | Kyutai Moshi backend surface using WebSocket and Opus framing. |
| `vona-transport-local` | Local HTTP/IPC transport helpers and length-prefixed CBOR framing. |
| `vona-sidecar` | Sidecar binary exposing Vona backends over HTTP and Unix-socket IPC. |
| `vona-test-harness` | Deterministic mock backend, scripted transport, fixtures, and benchmark harnesses. |

The workspace is backend-agnostic by design. Provider-specific integrations live in adapter crates; the `vona-core` crate stays focused on stable contracts, while `vona` is the crates.io facade for applications that want one dependency with opt-in features.

## Current Status

Vona is pre-1.0 and suitable for integration experiments, adapter development, and deterministic runtime testing. The public APIs may still change before a stable release.

Implemented today:

- step-oriented speech-to-speech backend trait
- event-stream realtime voice backend trait for hosted APIs, Moshi-family dialogue, and open realtime voice models
- audio transport trait
- session driver with metrics for first audio, tool calls, interruptions, and fallback decisions
- skill execution registry with schema validation and audit events
- context injection through `ExternalContextEvent`
- passthrough, Seamless M4T-style, Moshi, HTTP sidecar, and local IPC surfaces
- protocol crates for OpenAI Realtime, Gemini Live, Azure Voice Live/Speech, Qwen realtime voice, ElevenLabs TTS, and Deepgram STT/TTS
- local Ollama text generation through `vona-ollama`
- Apple Silicon MLX audio experiments through `vona-mlx`, `vona-mlx-whisper`, and `vona-mlx-qwen3-tts`
- local model provisioning manifests, explicit artifact downloads, and cache inspection for local model adapters
- deterministic realtime voice harness for tool-call, interruption, latency-mark, and event-order testing
- deterministic test harnesses and release-gate benchmarks

Known limits:

- production transport adapters such as LiveKit are not included yet
- the Seamless local ONNX path still needs operator-supplied model artifacts wired into a provisioning plan
- MLX speech loaders are experimental, Apple Silicon-focused, and require explicit local model artifacts
- Ollama text generation expects a reachable local Ollama server and an installed model such as `phi4-mini`
- cloud provider crates currently implement config and protocol mapping, not live credentialed CI tests
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

Native MLX speech builds require Apple Silicon, Xcode command line tools or Xcode, and the Metal compiler:

```bash
xcode-select --install
xcrun -f metal
```

For local release builds that exercise MLX kernels, prefer the host CPU tuning flag:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build -p vona --release --features "mlx-whisper-native mlx-qwen3-tts-native"
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

## Installation

For most applications, depend on the facade crate and enable the surfaces you need:

```bash
cargo add vona --features seamless,transport-local
```

Available facade features:

- `seamless`: re-export `vona-seamless`
- `moshi`: re-export `vona-moshi`
- `ollama`: re-export `vona-ollama`
- `mlx`: re-export `vona-mlx`
- `mlx-models-loader`: enable the optional `mlx-models` loader hook in `vona-mlx`
- `mlx-whisper`: re-export `vona-mlx-whisper`
- `mlx-qwen3-tts`: re-export `vona-mlx-qwen3-tts`
- `mlx-native`: enable native MLX support in `vona-mlx`
- `mlx-whisper-native`: enable native MLX support for the Whisper STT adapter
- `mlx-qwen3-tts-native`: enable native MLX support for the Qwen3 TTS adapter
- `transport-local`: re-export `vona-transport-local` and enable `seamless`
- `test-harness`: re-export `vona-test-harness`
- `openai-realtime`: re-export `vona-openai-realtime`
- `qwen`: re-export `vona-qwen`
- `gemini-live`: re-export `vona-gemini-live`
- `elevenlabs`: re-export `vona-elevenlabs`
- `deepgram`: re-export `vona-deepgram`
- `azure-speech`: re-export `vona-azure-speech`
- `model-provisioning`: re-export `vona-model-provisioning`
- `cloud`: enable the hosted cloud provider protocol/component crates
- `all`: enable every facade feature

You can also depend on lower-level crates directly:

```bash
cargo add vona-core
cargo add vona-seamless
cargo add vona-ollama
```

From a source checkout, use path dependencies:

```toml
[dependencies]
vona = { path = "crates/vona", features = ["seamless"] }
```

For local Ollama plus native MLX speech experiments from a source checkout:

```toml
[dependencies]
vona = { path = "crates/vona", features = ["ollama", "mlx-whisper-native", "mlx-qwen3-tts-native", "model-provisioning"] }
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

See [docs/sts-model-coverage.md](docs/sts-model-coverage.md) for how Vona distinguishes translation STS, full-duplex dialogue, hosted realtime APIs, open realtime voice models, and cascaded ASR+LLM+TTS systems.

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

## Local MLX And Ollama Benchmark

The facade includes an ignored-by-default local benchmark example that wires Qwen3 TTS, Whisper STT, and Ollama text generation together for 100 voice+chat cases. It requires local model artifacts and a running Ollama server:

```bash
ollama pull phi4-mini

export VONA_E2E_QWEN3_TTS_MODEL=/absolute/path/to/qwen3-tts
export VONA_E2E_WHISPER_MODEL=/absolute/path/to/distil-whisper
export VONA_E2E_OLLAMA_MODEL=phi4-mini

RUSTFLAGS="-C target-cpu=native" cargo run -p vona \
  --features "ollama mlx-whisper-native mlx-qwen3-tts-native model-provisioning" \
  --example mlx_ollama_voice_bench --locked
```

The historical 100-case run record lives in [docs/mlx-ollama-e2e-benchmark.md](docs/mlx-ollama-e2e-benchmark.md). It documents the benchmark shape and any quality caveats for that run.

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
- optional adapter facade feature checks
- native MLX compile checks on macOS when `xcrun metal` is available
- deterministic transport smoke benchmarks
- benchmark result generation in [docs/benchmark-results.md](docs/benchmark-results.md)

Read the full checklist in [docs/release-readiness-checklist.md](docs/release-readiness-checklist.md).

## Repository Layout

```text
crates/
  vona/                  facade crate with optional adapter features
  vona-core/             core runtime contracts
  vona-ollama/           local Ollama text generation adapter
  vona-mlx/              MLX audio engine facade
  vona-mlx-speech/       shared MLX speech loading utilities
  vona-mlx-whisper/      native MLX Whisper STT adapter
  vona-mlx-qwen3-tts/    native MLX Qwen3 TTS adapter
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

## Publishing

The crates are intended to publish in dependency order:

1. `vona-core`
2. `vona-model-provisioning`
3. `vona-ollama`
4. `vona-mlx-speech`
5. `vona-mlx`
6. `vona-mlx-whisper`
7. `vona-mlx-qwen3-tts`
8. `vona-openai-realtime`
9. `vona-gemini-live`
10. `vona-azure-speech`
11. `vona-elevenlabs`
12. `vona-deepgram`
13. `vona-qwen`
14. `vona-seamless`
15. `vona-moshi`
16. `vona-test-harness`
17. `vona-transport-local`
18. `vona-sidecar`
19. `vona`

The order matters because the facade crate depends on the adapter crates, and adapter crates depend on `vona-core`.

Use `scripts/release_crates.sh --release current|patch|minor|major` to update release metadata, run the release gate, package crates in order, and optionally publish with `--publish`.

## Security

Please do not open public issues for security vulnerabilities. Report them using the process in [SECURITY.md](SECURITY.md).

## License

Vona is licensed under the [MIT License](LICENSE).
