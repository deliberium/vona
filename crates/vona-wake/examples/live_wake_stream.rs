use serde::Serialize;
use serde_json::{Value, json};
use vona_core::types::AudioInputFrame;
use vona_wake::{
    EmbeddingSpeakerVerifier, EnergyWakeDetector, SpeakerProfile, WakeContext, WakeDecision,
    WakeGate, WakePolicy, simple_audio_embedding,
};

#[derive(Debug, Serialize)]
struct LiveWakeStreamReport {
    frames_streamed: usize,
    accepted: bool,
    accepted_at_sequence: Option<u64>,
    speaker_id: Option<String>,
    unauthorized_rejected: bool,
    privacy_suppressed: bool,
    final_state: String,
}

fn main() {
    let stream = synthetic_live_stream();
    let enrolled_voice = SpeakerProfile {
        speaker_id: "local-owner".to_string(),
        embedding: simple_audio_embedding(&stream),
        metadata: json!({"source": "live_wake_stream_example"}),
    };
    let happy = run_happy_path(&stream, enrolled_voice.clone());
    let unauthorized_rejected = run_unauthorized_path(&stream);
    let privacy_suppressed = run_privacy_path(&stream);

    let report = LiveWakeStreamReport {
        frames_streamed: stream.len(),
        accepted: happy.accepted,
        accepted_at_sequence: happy.accepted_at_sequence,
        speaker_id: happy.speaker_id,
        unauthorized_rejected,
        privacy_suppressed,
        final_state: happy.final_state,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
    assert!(report.accepted, "synthetic live stream was not admitted");
    assert_eq!(report.speaker_id.as_deref(), Some("local-owner"));
    assert!(
        report.unauthorized_rejected,
        "unauthorized speaker stream was not rejected"
    );
    assert!(
        report.privacy_suppressed,
        "privacy-mode stream was not suppressed"
    );
}

fn run_happy_path(
    stream: &[AudioInputFrame],
    enrolled_voice: SpeakerProfile,
) -> LiveWakeStreamReport {
    let mut gate = WakeGate::with_verifier(
        EnergyWakeDetector {
            phrase: Some("hey vona".to_string()),
            average_abs_threshold: 0.03,
            peak_abs_threshold: 0.08,
        },
        EmbeddingSpeakerVerifier,
        WakePolicy {
            wake_phrases: vec!["hey vona".to_string()],
            candidate_threshold: 0.25,
            accept_threshold: 0.55,
            require_speaker_verification: true,
            speaker_threshold: 0.92,
            preroll_ms: 500,
            ..WakePolicy::default()
        },
    );

    let context = WakeContext {
        allowed_speakers: vec![enrolled_voice],
        ..WakeContext::default()
    };

    let mut accepted = false;
    let mut accepted_at_sequence = None;
    let mut speaker_id = None;
    for frame in stream.iter().cloned() {
        if let WakeDecision::Accepted {
            speaker, preroll, ..
        } = gate.push_frame(frame.clone(), &context)
        {
            accepted = true;
            accepted_at_sequence = Some(frame.sequence);
            speaker_id = speaker.map(|speaker| speaker.speaker_id);
            assert!(
                !preroll.is_empty(),
                "live admission should include pre-roll frames"
            );
            break;
        }
    }

    LiveWakeStreamReport {
        frames_streamed: stream.len(),
        accepted,
        accepted_at_sequence,
        speaker_id,
        unauthorized_rejected: false,
        privacy_suppressed: false,
        final_state: format!("{:?}", gate.state()),
    }
}

fn run_unauthorized_path(stream: &[AudioInputFrame]) -> bool {
    let mut gate = WakeGate::with_verifier(
        EnergyWakeDetector {
            phrase: Some("hey vona".to_string()),
            average_abs_threshold: 0.03,
            peak_abs_threshold: 0.08,
        },
        EmbeddingSpeakerVerifier,
        WakePolicy {
            candidate_threshold: 0.25,
            accept_threshold: 0.55,
            require_speaker_verification: true,
            speaker_threshold: 0.99,
            ..WakePolicy::default()
        },
    );
    let context = WakeContext {
        allowed_speakers: vec![SpeakerProfile {
            speaker_id: "guest".to_string(),
            embedding: vec![0.0; 12],
            metadata: Value::Null,
        }],
        ..WakeContext::default()
    };
    stream.iter().cloned().any(|frame| {
        matches!(
            gate.push_frame(frame, &context),
            WakeDecision::Rejected { .. }
        )
    })
}

fn run_privacy_path(stream: &[AudioInputFrame]) -> bool {
    let mut gate = WakeGate::new(EnergyWakeDetector::default(), WakePolicy::default());
    let context = WakeContext {
        privacy_mode: true,
        ..WakeContext::default()
    };
    stream.iter().cloned().any(|frame| {
        matches!(
            gate.push_frame(frame, &context),
            WakeDecision::Suppressed { .. }
        )
    })
}

fn synthetic_live_stream() -> Vec<AudioInputFrame> {
    let mut sequence = 0_u64;
    let mut frames = Vec::new();
    for amplitude in [0.002, 0.003, 0.004, 0.006, 0.18, 0.22] {
        let samples = (0..320)
            .map(|index| {
                let carrier = ((index as f32) / 8.0).sin();
                carrier * amplitude
            })
            .collect::<Vec<_>>();
        frames.push(AudioInputFrame {
            sequence,
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        });
        sequence += 320;
    }
    frames
}
