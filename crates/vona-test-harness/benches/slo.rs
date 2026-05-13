//! Production SLO benchmarks for the core vona session pipeline.
//!
//! These benchmarks use the deterministic `MockBackend` and `ScriptedTransport`
//! so they run without any external services and produce stable, reproducible
//! numbers that can be tracked as release baselines.
//!
//! # SLO targets (documented)
//!
//! | Benchmark                        | Target (P99)  |
//! |----------------------------------|---------------|
//! | `backend_step_latency`           | < 500 µs      |
//! | `session_lifecycle`              | < 1 ms        |
//! | `transport_loopback_256_samples` | < 100 µs      |
//! | `inject_and_drain_10_events`     | < 200 µs      |
//!
//! Run with:
//! ```text
//! cargo bench -p vona-test-harness
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use vona::{
    AudioInputFrame, AudioOutputFrame, BackendStep, SessionConfig, SpeechToSpeechBackend,
    transport::AudioTransport,
};
use vona_test_harness::{MockBackend, ScriptedTransport};

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn dummy_input_frame(num_samples: usize) -> AudioInputFrame {
    AudioInputFrame {
        sequence: 1,
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0.0f32; num_samples],
    }
}

fn dummy_output_step(num_samples: usize) -> BackendStep {
    BackendStep {
        output_audio: vec![AudioOutputFrame {
            sequence: 1,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0f32; num_samples],
            is_filler: false,
        }],
        control_events: vec![],
        transcript: None,
        finished: false,
        debug_payload: None,
    }
}

/// Benchmark: single `step()` call through the mock backend.
///
/// SLO target: P99 < 500 µs (in-process, no I/O).
fn bench_backend_step_latency(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("backend_step");

    for num_samples in [160usize, 320, 960, 3_840] {
        group.bench_with_input(format!("{num_samples}_samples"), &num_samples, |b, &n| {
            b.iter_batched(
                || {
                    let backend = MockBackend::default();
                    backend.push_step(dummy_output_step(n));
                    let session = runtime
                        .block_on(backend.start_session(SessionConfig::default()))
                        .unwrap();
                    (backend, session, dummy_input_frame(n))
                },
                |(backend, mut session, frame)| {
                    runtime.block_on(async {
                        criterion::black_box(backend.step(&mut session, frame).await.unwrap())
                    })
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmark: full session lifecycle — `start_session` + one `step` + `end_session`.
///
/// SLO target: P99 < 1 ms.
fn bench_session_lifecycle(c: &mut Criterion) {
    let runtime = rt();

    c.bench_function("session_lifecycle", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let backend = MockBackend::default();
                backend.push_step(dummy_output_step(320));
                let mut session = backend
                    .start_session(SessionConfig::default())
                    .await
                    .unwrap();
                let _ = backend
                    .step(&mut session, dummy_input_frame(320))
                    .await
                    .unwrap();
                backend.end_session(session).await.unwrap();
            })
        })
    });
}

/// Benchmark: in-process transport loopback round-trip.
///
/// SLO target: P99 < 100 µs.
fn bench_transport_loopback(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("transport_loopback");

    for num_samples in [160usize, 320, 960] {
        group.bench_with_input(format!("{num_samples}_samples"), &num_samples, |b, &n| {
            b.iter_batched(
                || {
                    let transport = ScriptedTransport::default();
                    transport.push_input(AudioInputFrame {
                        sequence: 0,
                        sample_rate_hz: 16_000,
                        channels: 1,
                        samples: vec![0.0f32; n],
                    });
                    transport
                },
                |transport| {
                    runtime.block_on(async {
                        let frame = transport.recv_frame().await.unwrap().unwrap();
                        let out = AudioOutputFrame {
                            sequence: frame.sequence,
                            sample_rate_hz: frame.sample_rate_hz,
                            channels: frame.channels,
                            samples: frame.samples,
                            is_filler: false,
                        };
                        transport.send_frame(out).await.unwrap();
                        criterion::black_box(transport.sent_frames())
                    })
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmark: inject 10 events then drain them through `step()`.
///
/// SLO target: P99 < 200 µs.
fn bench_inject_and_drain_events(c: &mut Criterion) {
    use vona::ExternalContextEvent;

    let runtime = rt();

    c.bench_function("inject_and_drain_10_events", |b| {
        b.iter_batched(
            || {
                let backend = MockBackend::default();
                backend.push_step(dummy_output_step(320));
                let session = runtime
                    .block_on(backend.start_session(SessionConfig::default()))
                    .unwrap();
                (backend, session)
            },
            |(backend, mut session)| {
                runtime.block_on(async {
                    for i in 0..10u32 {
                        backend
                            .inject_event(
                                &mut session,
                                ExternalContextEvent {
                                    source: format!("vona.event_{i}"),
                                    spoken_summary: Some(format!("event {i}")),
                                    payload: serde_json::json!(null),
                                },
                            )
                            .await
                            .unwrap();
                    }
                    criterion::black_box(
                        backend
                            .step(&mut session, dummy_input_frame(320))
                            .await
                            .unwrap(),
                    )
                })
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    slo_benches,
    bench_backend_step_latency,
    bench_session_lifecycle,
    bench_transport_loopback,
    bench_inject_and_drain_events,
);
criterion_main!(slo_benches);
