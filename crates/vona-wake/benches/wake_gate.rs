use async_trait::async_trait;
use criterion::{Criterion, criterion_group, criterion_main};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use vona_core::transport::{AudioTransport, TransportError};
use vona_core::types::{AudioInputFrame, AudioOutputFrame};
use vona_wake::{EnergyWakeDetector, WakeContext, WakeGate, WakeGatedTransport, WakePolicy};

fn bench_wake_gate_push(c: &mut Criterion) {
    c.bench_function("wake_gate_push_energy_detector", |b| {
        b.iter(|| {
            let mut gate = WakeGate::new(
                EnergyWakeDetector {
                    average_abs_threshold: 0.05,
                    peak_abs_threshold: 0.2,
                    ..EnergyWakeDetector::default()
                },
                WakePolicy {
                    candidate_threshold: 0.4,
                    accept_threshold: 0.8,
                    preroll_ms: 800,
                    ..WakePolicy::default()
                },
            );
            let context = WakeContext::default();
            for sequence in 0..32 {
                let samples = if sequence == 31 {
                    vec![0.25; 320]
                } else {
                    vec![0.005; 320]
                };
                let _ = gate.push_frame(
                    AudioInputFrame {
                        sequence: sequence * 320,
                        sample_rate_hz: 16_000,
                        channels: 1,
                        samples,
                    },
                    &context,
                );
            }
        });
    });
}

fn bench_wake_gated_transport_admission(c: &mut Criterion) {
    let runtime = Runtime::new().expect("tokio runtime");
    c.bench_function("wake_gated_transport_admission", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let transport = BenchTransport::new(scripted_frames());
                let gate = WakeGate::new(
                    EnergyWakeDetector {
                        average_abs_threshold: 0.05,
                        peak_abs_threshold: 0.2,
                        ..EnergyWakeDetector::default()
                    },
                    WakePolicy {
                        candidate_threshold: 0.4,
                        accept_threshold: 0.8,
                        preroll_ms: 800,
                        ..WakePolicy::default()
                    },
                );
                let gated = WakeGatedTransport::new(transport, gate, WakeContext::default());
                let mut admitted = 0usize;
                while admitted < 4 {
                    if gated.recv_frame().await.expect("transport").is_some() {
                        admitted += 1;
                    } else {
                        break;
                    }
                }
                admitted
            })
        });
    });
}

fn scripted_frames() -> Vec<AudioInputFrame> {
    (0..36)
        .map(|sequence| {
            let samples = if sequence >= 31 {
                vec![0.25; 320]
            } else {
                vec![0.005; 320]
            };
            AudioInputFrame {
                sequence: sequence * 320,
                sample_rate_hz: 16_000,
                channels: 1,
                samples,
            }
        })
        .collect()
}

#[derive(Clone)]
struct BenchTransport {
    incoming: Arc<Mutex<VecDeque<AudioInputFrame>>>,
}

impl BenchTransport {
    fn new(frames: Vec<AudioInputFrame>) -> Self {
        Self {
            incoming: Arc::new(Mutex::new(frames.into())),
        }
    }
}

#[async_trait]
impl AudioTransport for BenchTransport {
    fn sample_rate_hz(&self) -> u32 {
        16_000
    }

    fn channels(&self) -> u16 {
        1
    }

    async fn recv_frame(&self) -> Result<Option<AudioInputFrame>, TransportError> {
        Ok(self.incoming.lock().expect("incoming").pop_front())
    }

    async fn send_frame(&self, _frame: AudioOutputFrame) -> Result<(), TransportError> {
        Ok(())
    }

    async fn clear_output(&self) -> Result<(), TransportError> {
        Ok(())
    }
}

criterion_group!(
    benches,
    bench_wake_gate_push,
    bench_wake_gated_transport_admission
);
criterion_main!(benches);
