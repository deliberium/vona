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

Do not use Vona if you need a turnkey assistant, hosted model service, audio
device stack, or production WebRTC integration out of the box. Vona includes
wake admission primitives through `vona-wake`, but applications still own
microphone permissions, enrollment, consent, profile storage, and product policy.

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
| `vona-moonshine` | Protected Moonshine local STT worker for fast Apple Silicon speech recognition. |
| `vona-mlx` | Apple Silicon MLX audio engine facade and streaming STT/TTS contracts. |
| `vona-mlx-speech` | Shared native Rust MLX speech model loading utilities. |
| `vona-mlx-whisper` | Native Rust MLX Whisper speech-to-text loader and inference surface. |
| `vona-mlx-qwen3-tts` | Native Rust MLX Qwen3 text-to-speech loader and inference surface. |
| `vona-seamless` | Seamless M4T-style local ONNX and HTTP sidecar backend adapters. |
| `vona-moshi` | Kyutai Moshi backend surface using WebSocket and Opus framing. |
| `vona-transport-local` | Local HTTP/IPC transport helpers and length-prefixed CBOR framing. |
| `vona-wake` | Wake admission, wake-gated transport, and optional speaker verification primitives. |
| `vona-sidecar` | Sidecar binary exposing Vona backends over HTTP and Unix-socket IPC. |
| `vona-test-harness` | Deterministic mock backend, scripted transport, fixtures, and benchmark harnesses. |

The workspace is backend-agnostic by design. Provider-specific integrations live in adapter crates; the `vona-core` crate stays focused on stable contracts, while `vona` is the crates.io facade for applications that want one dependency with opt-in features.

See [`docs/vona-wake.md`](docs/vona-wake.md) for the Vona Wake architecture,
speaker-gated admission model, evaluation harness, and real-corpus evidence
workflow.

See [`docs/moonshine-asr-benchmark.md`](docs/moonshine-asr-benchmark.md) for the
2,000-case generated-voice Moonshine ASR benchmark. The result is fast but not
standalone release-grade: 0.0887 average RTF and 0.301 average WER, with
human-recorded evidence still required before making reliability claims.

## Current Status

Vona is pre-1.0 and suitable for integration experiments, adapter development, and deterministic runtime testing. The public APIs may still change before a stable release.

Implemented today:

- step-oriented speech-to-speech backend trait
- event-stream realtime voice backend trait for hosted APIs, Moshi-family dialogue, and open realtime voice models
- audio transport trait
- session driver with metrics for first audio, tool calls, interruptions, and fallback decisions
- skill execution registry with schema validation and audit events
- context injection through `ExternalContextEvent`
- wake admission, template wake detection, wake-gated transport, and optional speaker verification primitives
- passthrough, Seamless M4T-style, Moshi, HTTP sidecar, and local IPC surfaces
- protocol crates for OpenAI Realtime, Gemini Live, Azure Voice Live/Speech, Qwen realtime voice, ElevenLabs TTS, and Deepgram STT/TTS
- local Ollama text generation through `vona-ollama`
- local Moonshine STT through a protected persistent worker with configurable transcript hotwords
- native Moonshine STT corpus tooling with manifest-driven generated-voice scoring and category rollups
- Apple Silicon MLX audio experiments through `vona-mlx`, `vona-mlx-whisper`, and `vona-mlx-qwen3-tts`
- provider-neutral realtime TTS policy for cached acknowledgements, Kokoro realtime speech, Piper low-power fallback, and Qwen3 premium synthesis
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
- `moonshine`: re-export `vona-moonshine`
- `mlx`: re-export `vona-mlx`
- `mlx-models-loader`: enable the optional `mlx-models` loader hook in `vona-mlx`
- `mlx-whisper`: re-export `vona-mlx-whisper`
- `mlx-qwen3-tts`: re-export `vona-mlx-qwen3-tts`
- `mlx-native`: enable native MLX support in `vona-mlx`
- `mlx-whisper-native`: enable native MLX support for the Whisper STT adapter
- `mlx-qwen3-tts-native`: enable native MLX support for the Qwen3 TTS adapter
- `transport-local`: re-export `vona-transport-local` and enable `seamless`
- `wake`: re-export `vona-wake`
- `test-harness`: re-export `vona-test-harness`
- `openai-realtime`: re-export `vona-openai-realtime`
- `qwen`: re-export `vona-qwen`
- `gemini-live`: re-export `vona-gemini-live`
- `kokoro-onnx`: re-export the phonemizer-backed Kokoro ONNX realtime TTS adapter
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

The facade includes an ignored-by-default local benchmark example that wires realtime TTS, Whisper STT, and Ollama text generation together for 100 voice+chat cases. It requires local model artifacts and a running Ollama server:

```bash
ollama pull phi4-mini

export VONA_E2E_KOKORO_ONNX_MODEL=/absolute/path/to/kokoro-0.onnx
export VONA_E2E_KOKORO_VOICES=/absolute/path/to/voices/0.bin
export VONA_E2E_WHISPER_MODEL=/absolute/path/to/distil-whisper
export VONA_E2E_OLLAMA_MODEL=phi4-mini
# Or compare multiple backend-profile LLMs on the same generated audio/transcript cases:
export VONA_E2E_OLLAMA_MODELS=phi4-mini,hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL
export VONA_E2E_MLX_VLM_MODELS=mlx-community/gemma-4-12B-it-4bit
export VONA_MLX_VLM_PYTHON=/absolute/path/to/python-with-mlx-vlm
export VONA_E2E_REALTIME_TTS_PROVIDER=kokoro
# Optional ASR vocabulary normalization for customer/project terms:
export VONA_WHISPER_HOTWORDS='Deliberium=delibrium|deliberiam,Gemma 4=gemma for|gemma four'
# Optional contained ASR runtime, recommended for release-grade native MLX use:
export VONA_E2E_WHISPER_RUNTIME=worker
export VONA_WHISPER_WORKER_BIN=/absolute/path/to/vona_mlx_whisper_worker

RUSTFLAGS="-C target-cpu=native" cargo run -p vona \
  --features "kokoro-onnx ollama mlx-whisper-native mlx-qwen3-tts-native model-provisioning" \
  --example mlx_ollama_voice_bench --locked
```

The benchmark routes synthesis through `PolicyAudioSynthesizer`. `VONA_E2E_REALTIME_TTS_PROVIDER` defaults to `kokoro`; it can also be set to `qwen3` or a custom provider name registered by the harness/application. `VONA_E2E_OLLAMA_MODELS` and `VONA_E2E_MLX_VLM_MODELS` run comma-separated A/B sets against the same TTS and STT cases, report per-model success/error counts, first-frame latency, and load/error rows, and keep model-load failures visible in the generated Markdown instead of aborting the run. The MLX-VLM path uses a persistent JSON-lines worker so the model remains loaded between turns, streams token deltas back to Vona as `TextGenerationFrame`s, and cooperatively cancels in-flight generation when the downstream stream is dropped; override the bundled worker script with `VONA_MLX_VLM_WORKER_SCRIPT` when needed.

Native Whisper postprocessing supports configurable transcript hotwords through `WhisperSpeechConfig::with_hotwords(...)` or `VONA_WHISPER_HOTWORDS`. The environment format is comma-separated `replacement=variant|variant` entries. Vona ships a small default list for local-stack terms such as Vona, Qwen, Ollama, and Whisper; downstream applications can replace it with their own customer vocabulary.

Native MLX inference is fast but not process-contained: if the underlying MLX C++/Metal runtime throws across the Rust FFI boundary, Rust cannot catch it and the process aborts. Release-grade deployments should use `ProtectedWhisperTranscriber` with the `vona_mlx_whisper_worker` binary. The parent process sends raw little-endian `f32` PCM behind a JSON-lines header, keeps one model-loaded worker warm, and restarts the worker after EOF/write/read failures instead of dying with the native MLX runtime.

Moonshine STT can run through a direct native binding or the same protected-worker idea without tying ASR to MLX Whisper. Enable the `moonshine` facade feature and create `NativeMoonshineTranscriber` with `VONA_MOONSHINE_LIBRARY_PATH` plus `VONA_MOONSHINE_MODEL_PATH`, or `ProtectedMoonshineTranscriber` with `VONA_MOONSHINE_PYTHON` and `VONA_MOONSHINE_WORKER_SCRIPT`. Both paths use raw little-endian `f32` PCM and the same hotword postprocessing. The current labeled synthetic benchmark favors `MEDIUM_STREAMING`: 12 cases, avg WER `0.019`, avg STT `392.1 ms`, near-exact `11/12`.

Build the worker with:

```bash
cargo build -p vona-mlx-whisper --features native-mlx --bin vona_mlx_whisper_worker --release
```

`mlx-sys` 0.2.0 builds its bundled `mlx-c` project, and that CMake project may fetch `ml-explore/mlx` from GitHub during compilation. For bootstrap/provisioning, prefetch MLX source with:

```bash
MLX_REF=main MLX_CACHE_DIR=resources/vendor/mlx ./scripts/prefetch_mlx_artifacts.sh
export MLX_SOURCE_DIR=resources/vendor/mlx
export METAL_CPP_SOURCE_DIR=resources/vendor/metal-cpp
export JSON_SOURCE_DIR=resources/vendor/nlohmann-json
export FMT_SOURCE_DIR=resources/vendor/fmt
export GGUF_SOURCE_DIR=resources/vendor/gguf-tools
```

Vona patches `mlx-sys` through Cargo `[patch.crates-io]` so native builds can pass `MLX_SOURCE_DIR`, `METAL_CPP_SOURCE_DIR`, `JSON_SOURCE_DIR`, `FMT_SOURCE_DIR`, and `GGUF_SOURCE_DIR` through to the bundled `mlx-c`/MLX CMake projects. The `resources/vendor/*` artifact directories are intentionally ignored by git because they are upstream source/header checkouts; bootstrap should prefetch them on machines that build native MLX artifacts.

The historical 100-case run record lives in [docs/mlx-ollama-e2e-benchmark.md](docs/mlx-ollama-e2e-benchmark.md). It documents the benchmark shape and any quality caveats for that run.

## Realtime TTS Policy

Vona separates voice-turn policy from individual TTS engines. The default local realtime policy is:

- cached acknowledgement audio for tiny nudges such as "Okay" or "One moment"
- Kokoro-82M ONNX for short realtime replies
- Piper for low-power mode and fallback
- Qwen3 TTS for long or premium-quality synthesis

Applications can use `RealtimeTtsPolicy` to select a provider, or `PolicyAudioSynthesizer` to route synthesis through configured provider slots. The default realtime provider is Kokoro, but downstream clients can replace it with `TtsProviderId::custom_realtime("provider-name")` and register any streaming-capable `AudioSynthesizer` through `with_custom_realtime_provider(...)`. The policy keeps Kokoro, Piper, Qwen3, and client-provided realtime TTS engines behind the same contract so downstream apps can change providers without rewriting voice-turn orchestration.

The `kokoro-onnx` feature provides `KokoroOnnxSynthesizer`, a phonemizer-backed Kokoro ONNX adapter that implements `AudioSynthesizer` and can be registered directly in the Kokoro realtime slot.

Model provisioning exposes provider-managed manifests for the realtime policy:

- `kokoro_82m_onnx_realtime_manifest()`
- `piper_low_power_tts_manifest()`
- `qwen3_tts_12hz_0_6b_base_bf16_manifest()`

## Local Text Routing Policy

Vona also separates text-model routing from individual LLM adapters. The default local policy is designed for the Gemma 4 versus phi4 benchmark result:

- `phi4-mini` through Ollama for short, latency-sensitive voice turns
- `mlx-community/gemma-4-12B-it-4bit` through the persistent MLX-VLM worker for long, complex, or reasoning-heavy turns
- explicit downstream override when an application wants to force a configured backend

Applications can use `TextRoutingPolicy` to inspect the selected backend, or `PolicyTextGenerator` to register both generators and route calls through one `TextGenerator` implementation. `TextRoutingRequest::interactive(...)` is the low-latency default; `TextRoutingRequest::reasoning(...)`, `expect_long_answer`, `prefer_reasoning_quality`, long prompts, or complex terms such as "compare", "debug", "architecture", "why", and "tradeoff" route to the reasoning backend. Downstream applications remain free to replace either side with `TextBackendId::custom("provider-name")`.

## Qwen3 TTS Streaming A/B Harness

The Qwen3 TTS crate keeps full offline synthesis as the reference oracle for streaming work. Use the A/B harness to compare that oracle against the selected streaming vocoder mode with the same text and deterministic seed:

```bash
export VONA_QWEN3_TTS_SEED=7
export VONA_MLX_QWEN3_TTS_STREAM_VOCODER_MODE=rolling
export VONA_MLX_QWEN3_TTS_STREAM_WINDOW_FRAMES=160
export VONA_QWEN3_TTS_AB_REPORT=target/qwen3-tts-ab.json

RUSTFLAGS="-C target-cpu=native" cargo run -p vona-mlx-qwen3-tts \
  --features native-mlx \
  --example qwen3_tts_ab_harness --locked -- \
  /absolute/path/to/qwen3-tts \
  "Hello from Vona. This is the streaming vocoder comparison."
```

The report includes time-to-first-audio, total offline and streaming synthesis time, waveform RMSE/MAE/max-delta/correlation, duration delta, and log-spectral RMSE. Set `VONA_QWEN3_TTS_AB_ENFORCE=1` to fail the run when quality thresholds are missed. The default thresholds can be overridden with `VONA_QWEN3_TTS_AB_MAX_RMSE`, `VONA_QWEN3_TTS_AB_MIN_CORRELATION`, and `VONA_QWEN3_TTS_AB_MAX_LOG_SPECTRAL_RMSE`.

Use `--suite` instead of a text argument to run the built-in six-case text suite:

```bash
VONA_QWEN3_TTS_AB_ENFORCE=1 cargo run -p vona-mlx-qwen3-tts \
  --features native-mlx \
  --example qwen3_tts_ab_harness --locked -- \
  /absolute/path/to/qwen3-tts --suite
```

Streaming vocoder modes are selected with `VONA_MLX_QWEN3_TTS_STREAM_VOCODER_MODE`:

- `rolling`: rolling-window overlap-add fallback, intended as the robust default while cached-state work is being proven.
- `prefix`: full-prefix re-vocoding oracle path, useful for debugging streaming correctness but usually too expensive for production.
- `cached`: experimental incremental waveform decoder/cache path. It requires `VONA_MLX_QWEN3_TTS_ENABLE_EXPERIMENTAL_CACHED_STATE=1`; keep it off for production until it passes the A/B harness with a material timing win.

The rolling mode window can be tuned with `VONA_MLX_QWEN3_TTS_STREAM_WINDOW_FRAMES`. The default is 160 frames because the six-case suite passed at 160, while 48, 64, 80, 96, and 128 frames failed at least one long or numeric case. Smaller windows can improve first-audio timing on short utterances, but should remain experimental until a broader suite passes.

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
  vona-wake/             wake admission, template detection, and optional speaker verification
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
2. `vona-wake`
3. `vona-model-provisioning`
4. `vona-ollama`
5. `vona-mlx-speech`
6. `vona-mlx`
7. `vona-mlx-whisper`
8. `vona-mlx-qwen3-tts`
9. `vona-openai-realtime`
10. `vona-gemini-live`
11. `vona-azure-speech`
12. `vona-elevenlabs`
13. `vona-deepgram`
14. `vona-qwen`
15. `vona-seamless`
16. `vona-moshi`
17. `vona-test-harness`
18. `vona-transport-local`
19. `vona-sidecar`
20. `vona`

The order matters because the facade crate depends on the adapter crates, and adapter crates depend on `vona-core`.

Use `scripts/release_crates.sh --release current|patch|minor|major` to update release metadata, run the release gate, package crates in order, and optionally publish with `--publish`.

## Security

Please do not open public issues for security vulnerabilities. Report them using the process in [SECURITY.md](SECURITY.md).

## License

Vona is licensed under the [MIT License](LICENSE).
