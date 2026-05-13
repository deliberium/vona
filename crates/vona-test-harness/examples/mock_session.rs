use std::sync::Arc;

use serde_json::json;
use vona::{
    AudioInputFrame, AudioOutputFrame, BackendStep, ControlEvent, FillerStrategy, SessionConfig,
    SkillCall, VonaRuntime, run_session,
};
use vona_test_harness::{AllowAllPolicy, EchoSkillExecutor, MockBackend, ScriptedTransport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let transport = ScriptedTransport::default();
    transport.push_input(AudioInputFrame {
        sequence: 1,
        sample_rate_hz: 24_000,
        channels: 1,
        samples: vec![0.0; 320],
    });

    let backend = MockBackend::default();
    backend.push_step(BackendStep {
        output_audio: vec![AudioOutputFrame {
            sequence: 1,
            sample_rate_hz: 24_000,
            channels: 1,
            samples: vec![0.25; 320],
            is_filler: false,
        }],
        control_events: vec![
            ControlEvent::SkillCall(SkillCall {
                name: "lookup_context".to_string(),
                arguments: json!({"topic": "release-readiness"}),
            }),
            ControlEvent::Interruption {
                reason: Some("barge_in".to_string()),
            },
        ],
        transcript: Some("mock transcript".to_string()),
        finished: true,
        debug_payload: Some(json!({"backend": "mock"})),
    });

    let runtime = VonaRuntime::new(
        Arc::new(EchoSkillExecutor),
        Arc::new(AllowAllPolicy),
        FillerStrategy::None,
    );

    let summary = run_session(
        transport.clone(),
        &backend,
        &runtime,
        SessionConfig {
            session_id: "mock-session-1".to_string(),
            ..SessionConfig::default()
        },
    )
    .await?;

    println!("session_id={}", summary.session_id);
    println!("close_reason={:?}", summary.close_reason);
    println!(
        "metrics time_to_first_audio_ms={:?} tool_calls={} interruptions={} fallback_count={}",
        summary.metrics.time_to_first_audio_ms,
        summary.metrics.tool_calls,
        summary.metrics.interruptions,
        summary.metrics.fallback_count
    );
    println!(
        "output_frames_after_interruption={}",
        transport.sent_frames().len()
    );
    println!("injected_events={}", backend.injected_events().len());

    Ok(())
}
