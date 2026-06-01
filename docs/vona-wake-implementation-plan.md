# Vona Wake Implementation Plan

`vona-wake` is the wake admission layer for Vona applications. It does not replace
speech-to-speech backends, application session policy, microphone ownership, or
user identity management. It decides when an audio stream should be admitted into
a Vona session and returns the evidence needed by the embedding application.

## Goals

- Provide a Rust-native wake admission crate that downstream applications can use
  instead of embedding OpenWakeWord-specific code in product surfaces.
- Keep OpenWakeWord available downstream as a configurable legacy/provider option,
  but make it disabled by default once `vona-wake` is wired.
- Support optional speaker verification for applications that want wake access
  limited to pre-enrolled users.
- Preserve Vona's provider-neutral design: wake policy sits before
  `AudioTransport`, `run_session`, realtime backends, and STS backends.
- Make the core testable without model assets, microphone hardware, or network
  access.

## Architecture

```text
application microphone/device layer
  -> vona-wake WakeGate
  -> accepted wake event with pre-roll and verification evidence
  -> Vona AudioTransport / realtime input / STS session
  -> application tools, policy, and UI
```

`vona-wake` owns:

- ring buffering and pre-roll extraction
- wake state machine
- acoustic/context admission policy
- detector and verifier traits
- wake metrics and decision evidence
- a `WakeGatedTransport<T, D, V>` adapter for Vona's `AudioTransport`

Downstream applications own:

- microphone permissions and device selection
- user identity, consent, enrollment, deletion, and profile storage
- allowed-user policy for the current device/workspace/room
- UI state and wake telemetry persistence
- OpenWakeWord legacy configuration
- model asset provisioning for any model-backed detector implementation

## Wake Pipeline

The intended production cascade is:

```text
cheap acoustic sentinels
  -> candidate phrase detector
  -> phrase/speaker verifier
  -> application authorization policy
  -> Vona session admission
```

The first crate version includes deterministic, asset-free implementations for
testing and downstream integration:

- `EnergyWakeDetector`: accepts speech-like frames using average/peak energy.
- `TemplateWakeDetector`: compares recent audio fingerprints with
  application-enrolled wake phrase templates.
- `NoopSpeakerVerifier`: always skips identity checks.
- `EmbeddingSpeakerVerifier`: compares caller-supplied embeddings with cosine
  similarity. Production applications can replace this with ONNX, MLX, or remote
  enrollment-backed implementations.

Model-backed detectors should remain optional and implement `WakeDetector` or
`SpeakerVerifier` without adding inference dependencies to `vona-wake` core.
Separate ONNX, MLX, or OpenWakeWord compatibility crates should only be added
when they wrap real model assets with stable IO contracts.

## Public Surface

Core types:

- `WakePolicy`: thresholds, phrases, pre-roll, cooldown, follow-up, barge-in, and
  optional speaker verification controls.
- `WakeContext`: application-supplied runtime context such as playback activity,
  privacy mode, follow-up eligibility, and allowed speaker profiles.
- `WakeDetector`: pluggable candidate detector.
- `SpeakerVerifier`: optional speaker identity primitive.
- `WakeGate`: stateful admission engine.
- `WakeDecision`: `Idle`, `Candidate`, `Accepted`, `Rejected`, or `Suppressed`.
- `WakeGatedTransport`: adapter that withholds microphone frames until wake
  admission succeeds, then releases pre-roll and live audio into Vona.

## Downstream Migration Notes

A downstream application that already owns wake handling can migrate to
`vona-wake` without handing user policy, enrollment, or storage to Vona.

Migration phases:

1. Add `vona-wake` to the application's Vona dependency features.
2. Make `vona-wake` the default wake admission provider.
3. Keep model-backed or legacy wake providers available as explicit opt-in
   choices when required.
4. Add an application adapter that maps product configuration into
   `WakePolicy` and `WakeContext`.
5. Route wake decisions through existing application metrics, playback
   suppression, follow-up mode, and active-turn admission.
6. Add optional voice verification:
   - a product-level voice verification enable flag
   - a speaker similarity threshold
   - application-managed allowed speaker IDs
   - application-managed encrypted profile storage
7. Keep enrollment and deletion in the downstream application, not in
   `vona-wake`.
8. Add downstream integration tests using deterministic detectors and speaker
   verifiers before introducing model assets.

## Testing And Validation

`vona-wake` must include:

- unit tests for state transitions, pre-roll, cooldown, follow-up, suppression, and
  thresholding
- speaker verification tests for allowed, rejected, and optional identity modes
- transport tests showing frames are withheld before wake and released after wake
- Criterion benchmarks for gate push latency and transport admission overhead
- automated live-stream harness covering accepted wake, speaker rejection, and
  privacy suppression

Downstream application validation should include:

- config tests proving `vona-wake` is default and legacy providers are opt-in
- runtime tests proving Vona wake decisions drive the existing wake detection flow
- voice verification tests proving unauthorized speakers do not start sessions
- live harness support using the application's existing turn/admission harness
  with `vona-wake` selected

## Completion Criteria

- `vona-wake` is a workspace crate and optional `vona` facade feature.
- The crate builds, formats, and passes tests.
- The benchmark harness runs locally.
- The automated live-stream harness runs locally.
- At least one downstream integration compiles against the new crate or has a
  checked-in migration patch/plan if the downstream workspace is not writable in
  the implementation session.
- Legacy wake providers remain configurable downstream but are no longer the
  default wake provider in Vona-owned examples.
