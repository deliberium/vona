//! Deterministic realtime/provider/provisioning benchmarks for Vona.
//!
//! These benchmarks intentionally stay network-free. They measure the protocol
//! and runtime surfaces that host applications exercise before a provider
//! WebSocket, sidecar process, or model server gets involved.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use vona_core::{
    AudioInputFrame, ExternalContextEvent, RealtimeVoiceBackend, RealtimeVoiceCapabilities,
    RealtimeVoiceInput, RealtimeVoiceOutput, RealtimeVoiceSessionConfig,
};
use vona_gemini_live::{GeminiLiveConfig, input_to_client_message as gemini_input_to_message};
use vona_model_provisioning::{
    HttpModelProvisioner, LocalModelProvider, ModelArtifact, ModelCache, ModelManifest,
    validate_manifest,
};
use vona_openai_realtime::{OpenAiRealtimeConfig, input_to_client_event as openai_input_to_event};
use vona_test_harness::ScriptedRealtimeBackend;

const BENCH_ARTIFACT_BYTES: &[u8] = b"vona benchmark artifact";
const BENCH_ARTIFACT_SHA256: &str =
    "69653f4444c8dfa3baab41c389e49f58e2e1538eb4c743c00797c1bd319b8c48";

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn audio_frame(samples: usize, sample_rate_hz: u32) -> AudioInputFrame {
    AudioInputFrame {
        sequence: 1,
        sample_rate_hz,
        channels: 1,
        samples: (0..samples)
            .map(|idx| ((idx % 64) as f32 / 64.0) - 0.5)
            .collect(),
    }
}

fn realtime_backend() -> ScriptedRealtimeBackend {
    let backend = ScriptedRealtimeBackend::with_capabilities(RealtimeVoiceCapabilities {
        supports_full_duplex: true,
        supports_streaming_audio_input: true,
        supports_streaming_audio_output: true,
        supports_tool_calls: true,
        supports_interruption: true,
        supports_context_injection: true,
        is_hosted_service: true,
        max_input_chunk_ms: Some(40),
    });
    backend.push_output(RealtimeVoiceOutput::TranscriptFragment {
        text: "ready".to_string(),
        final_fragment: false,
    });
    backend.push_output(RealtimeVoiceOutput::Interruption {
        reason: Some("barge_in".to_string()),
    });
    backend.push_output(RealtimeVoiceOutput::ResponseCompleted {
        reason: Some("response.done".to_string()),
    });
    backend
}

fn bench_scripted_realtime_event_flow(c: &mut Criterion) {
    let runtime = rt();

    c.bench_function("scripted_realtime_event_flow", |b| {
        b.iter_batched(
            || {
                let backend = realtime_backend();
                let session = runtime
                    .block_on(backend.start_realtime_session(RealtimeVoiceSessionConfig::default()))
                    .unwrap();
                (backend, session)
            },
            |(backend, mut session)| {
                runtime.block_on(async {
                    backend
                        .send_realtime_event(
                            &mut session,
                            RealtimeVoiceInput::Audio(audio_frame(320, 24_000)),
                        )
                        .await
                        .unwrap();
                    backend
                        .send_realtime_event(
                            &mut session,
                            RealtimeVoiceInput::ToolResult(ExternalContextEvent {
                                source: "skill:lookup_context".to_string(),
                                spoken_summary: Some("context ready".to_string()),
                                payload: serde_json::json!({"ok": true}),
                            }),
                        )
                        .await
                        .unwrap();
                    while backend
                        .recv_realtime_event(&mut session)
                        .await
                        .unwrap()
                        .is_some()
                    {}
                    criterion::black_box(backend.received_inputs())
                })
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_provider_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("provider_mapping");
    let openai_frame = audio_frame(320, 24_000);
    let gemini_frame = audio_frame(320, 16_000);
    let openai_audio = RealtimeVoiceInput::Audio(openai_frame);
    let gemini_audio = RealtimeVoiceInput::Audio(gemini_frame);
    let tool_result = RealtimeVoiceInput::ToolResult(ExternalContextEvent {
        source: "lookup_context".to_string(),
        spoken_summary: Some("context ready".to_string()),
        payload: serde_json::json!({"answer": "ready"}),
    });

    group.bench_function("openai_audio_append_20ms", |b| {
        b.iter(|| {
            criterion::black_box(openai_input_to_event(criterion::black_box(
                openai_audio.clone(),
            )))
        })
    });
    group.bench_function("openai_tool_result", |b| {
        b.iter(|| {
            criterion::black_box(openai_input_to_event(criterion::black_box(
                tool_result.clone(),
            )))
        })
    });
    group.bench_function("gemini_audio_message_20ms", |b| {
        b.iter(|| {
            criterion::black_box(gemini_input_to_message(
                criterion::black_box(gemini_audio.clone()),
                16_000,
            ))
        })
    });
    group.bench_function("session_config_openai", |b| {
        let config = OpenAiRealtimeConfig::default();
        b.iter(|| criterion::black_box(config.session_config("bench-session")))
    });
    group.bench_function("session_config_gemini", |b| {
        let config = GeminiLiveConfig::default();
        b.iter(|| criterion::black_box(config.session_config("bench-session")))
    });
    group.finish();
}

fn manifest() -> ModelManifest {
    ModelManifest {
        id: "bench/model".to_string(),
        provider: LocalModelProvider::Custom {
            name: "bench".to_string(),
        },
        artifacts: vec![ModelArtifact {
            name: "artifact".to_string(),
            relative_path: PathBuf::from("artifact.bin"),
            source_url: None,
            expected_size_bytes: Some(BENCH_ARTIFACT_BYTES.len() as u64),
            sha256: Some(BENCH_ARTIFACT_SHA256.to_string()),
        }],
    }
}

fn temp_cache() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    std::env::temp_dir().join(format!("vona-bench-cache-{nanos}"))
}

fn bench_model_provisioning(c: &mut Criterion) {
    let runtime = rt();
    let manifest = manifest();

    c.bench_function("model_manifest_validate_and_plan", |b| {
        b.iter_batched(
            || ModelCache { root: temp_cache() },
            |cache| {
                validate_manifest(&manifest).unwrap();
                criterion::black_box(cache.inspect(&manifest))
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("model_provisioning_verify_present_artifact", |b| {
        b.iter_batched(
            || {
                let cache = ModelCache { root: temp_cache() };
                let path = cache.artifact_path(&manifest, &manifest.artifacts[0]);
                runtime.block_on(async {
                    tokio::fs::create_dir_all(path.parent().unwrap())
                        .await
                        .unwrap();
                    let mut file = tokio::fs::File::create(&path).await.unwrap();
                    file.write_all(BENCH_ARTIFACT_BYTES).await.unwrap();
                    file.flush().await.unwrap();
                });
                cache
            },
            |cache| {
                runtime.block_on(async {
                    let plan = HttpModelProvisioner::default()
                        .provision_missing(&cache, &manifest)
                        .await
                        .unwrap();
                    let _ = tokio::fs::remove_dir_all(&cache.root).await;
                    criterion::black_box(plan)
                })
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    realtime_benches,
    bench_scripted_realtime_event_flow,
    bench_provider_mapping,
    bench_model_provisioning,
);
criterion_main!(realtime_benches);
