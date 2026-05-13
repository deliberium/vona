//! Throughput benchmarks for the `resample_mono` function.
//!
//! # SLO targets (documented)
//!
//! | Benchmark                      | Target throughput |
//! |--------------------------------|-------------------|
//! | `resample_8k_to_16k_1s`        | > 50 × real-time  |
//! | `resample_48k_to_16k_1s`       | > 30 × real-time  |
//! | `resample_16k_identity_1s`     | > 500 × real-time |
//!
//! Run with:
//! ```text
//! cargo bench -p vona-seamless
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vona_seamless::onnx_runtime::resample_mono;

/// Generates a mono sine wave at `freq_hz` with `num_samples` samples.
fn sine_wave(num_samples: usize, freq_hz: f32, sample_rate_hz: u32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate_hz as f32).sin())
        .collect()
}

/// Benchmark: resample_mono for common rate conversions, 1-second buffers.
fn bench_resample_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_mono");

    let scenarios: &[(&str, u32, u32)] = &[
        ("8k_to_16k", 8_000, 16_000),
        ("16k_identity", 16_000, 16_000),
        ("24k_to_16k", 24_000, 16_000),
        ("48k_to_16k", 48_000, 16_000),
        ("16k_to_48k", 16_000, 48_000),
    ];

    for &(label, src_hz, dst_hz) in scenarios {
        let num_samples = src_hz as usize; // 1 second of audio
        let input = sine_wave(num_samples, 440.0, src_hz);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(src_hz, dst_hz, &input),
            |b, (src, dst, samples)| {
                b.iter(|| {
                    criterion::black_box(resample_mono(criterion::black_box(samples), *src, *dst))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: resample with short buffers (typical streaming frame sizes).
fn bench_resample_mono_frame_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_mono_frames");

    // 10 ms, 20 ms, 40 ms frames at 16 kHz → 48 kHz (common sidecar upsample path)
    for &frame_ms in &[10usize, 20, 40] {
        let src_hz: u32 = 16_000;
        let dst_hz: u32 = 48_000;
        let num_samples = src_hz as usize * frame_ms / 1_000;
        let input = sine_wave(num_samples, 440.0, src_hz);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{frame_ms}ms")),
            &(src_hz, dst_hz, &input),
            |b, (src, dst, samples)| {
                b.iter(|| {
                    criterion::black_box(resample_mono(criterion::black_box(samples), *src, *dst))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    resample_benches,
    bench_resample_mono,
    bench_resample_mono_frame_sizes
);
criterion_main!(resample_benches);
