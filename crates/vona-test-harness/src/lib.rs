use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vona::backend::{BackendCapabilities, BackendError, BackendStep, SpeechToSpeechBackend};
use vona::runtime::{FallbackReason, SessionPolicy};
use vona::session::SessionConfig;
use vona::skills::{SkillError, SkillExecutor, SkillOutput};
use vona::transport::{AudioTransport, TransportError};
use vona::types::{
    AudioInputFrame, AudioOutputFrame, ControlEvent, ExternalContextEvent, SkillCall, SkillContext,
};

#[derive(Clone, Default)]
pub struct ScriptedTransport {
    incoming: Arc<Mutex<VecDeque<AudioInputFrame>>>,
    sent: Arc<Mutex<Vec<AudioOutputFrame>>>,
}

impl ScriptedTransport {
    pub fn push_input(&self, frame: AudioInputFrame) {
        self.incoming
            .lock()
            .expect("incoming queue")
            .push_back(frame);
    }

    pub fn sent_frames(&self) -> Vec<AudioOutputFrame> {
        self.sent.lock().expect("sent frames").clone()
    }
}

#[async_trait]
impl AudioTransport for ScriptedTransport {
    fn sample_rate_hz(&self) -> u32 {
        24_000
    }

    fn channels(&self) -> u16 {
        1
    }

    async fn recv_frame(&self) -> Result<Option<AudioInputFrame>, TransportError> {
        Ok(self.incoming.lock().expect("incoming queue").pop_front())
    }

    async fn send_frame(&self, frame: AudioOutputFrame) -> Result<(), TransportError> {
        self.sent.lock().expect("sent frames").push(frame);
        Ok(())
    }

    async fn clear_output(&self) -> Result<(), TransportError> {
        self.sent.lock().expect("sent frames").clear();
        Ok(())
    }
}

#[derive(Default)]
pub struct MockBackend {
    steps: Arc<Mutex<VecDeque<BackendStep>>>,
    injections: Arc<Mutex<Vec<ExternalContextEvent>>>,
}

impl MockBackend {
    pub fn push_step(&self, step: BackendStep) {
        self.steps.lock().expect("backend steps").push_back(step);
    }

    pub fn injected_events(&self) -> Vec<ExternalContextEvent> {
        self.injections.lock().expect("injections").clone()
    }
}

#[async_trait]
impl SpeechToSpeechBackend for MockBackend {
    type Session = SessionConfig;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_context_injection: true,
            ..BackendCapabilities::default()
        }
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(config)
    }

    async fn step(
        &self,
        _session: &mut Self::Session,
        _input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        Ok(self
            .steps
            .lock()
            .expect("backend steps")
            .pop_front()
            .unwrap_or_default())
    }

    async fn inject_event(
        &self,
        _session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        self.injections.lock().expect("injections").push(event);
        Ok(())
    }

    async fn end_session(&self, _session: Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}

pub struct EchoSkillExecutor;

#[async_trait]
impl SkillExecutor for EchoSkillExecutor {
    async fn execute(
        &self,
        call: SkillCall,
        _context: SkillContext,
    ) -> Result<SkillOutput, SkillError> {
        Ok(SkillOutput {
            spoken_summary: format!("executed {}", call.name),
            structured_payload: Some(json!({"name": call.name, "args": call.arguments})),
            audit_payload: None,
        })
    }
}

pub struct AllowAllPolicy;

impl SessionPolicy for AllowAllPolicy {
    fn should_accept_control_event(&self, _event: &ControlEvent) -> bool {
        true
    }

    fn should_fallback_to_bridge(&self, _reason: &FallbackReason) -> bool {
        false
    }

    fn max_tool_latency_ms(&self) -> u64 {
        500
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WaveformFixture {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

pub fn workspace_fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

pub fn load_waveform_fixture(name: &str) -> WaveformFixture {
    let path = workspace_fixture_path(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read waveform fixture {}: {err}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!(
            "failed to decode waveform fixture {}: {err}",
            path.display()
        )
    })
}

pub async fn measure_loopback_latency_ms(transport: &ScriptedTransport) -> u128 {
    let started = Instant::now();
    let inbound = transport
        .recv_frame()
        .await
        .expect("receive frame")
        .expect("queued frame");
    transport
        .send_frame(AudioOutputFrame {
            sequence: inbound.sequence,
            sample_rate_hz: inbound.sample_rate_hz,
            channels: inbound.channels,
            samples: inbound.samples,
            is_filler: false,
        })
        .await
        .expect("send frame");
    started.elapsed().as_millis()
}

#[tokio::test]
async fn runtime_executes_skill_call_and_returns_injection_event() {
    use vona::runtime::{FillerStrategy, RuntimeDecision, VonaRuntime};

    let runtime = VonaRuntime::new(
        Arc::new(EchoSkillExecutor),
        Arc::new(AllowAllPolicy),
        FillerStrategy::StaticClip,
    );
    let context = SkillContext {
        session_id: "session-1".to_string(),
        user_id: Some("user-1".to_string()),
        thread_id: Some("thread-1".to_string()),
        metadata: json!({"surface": "test"}),
    };

    let decision = runtime
        .handle_control_event(
            &ControlEvent::SkillCall(SkillCall {
                name: "get_weather".to_string(),
                arguments: json!({"city": "Nairobi"}),
            }),
            context,
        )
        .await
        .expect("skill decision");

    match decision {
        RuntimeDecision::InjectContext(event) => {
            assert_eq!(event.source, "skill:get_weather");
            assert_eq!(
                event.spoken_summary.as_deref(),
                Some("executed get_weather")
            );
        }
        other => panic!("unexpected runtime decision: {other:?}"),
    }
}

#[tokio::test]
async fn waveform_fixture_round_trips_through_scripted_transport() {
    let fixture = load_waveform_fixture("sine-16khz-mono.json");
    let transport = ScriptedTransport::default();
    transport.push_input(AudioInputFrame {
        sequence: 1,
        sample_rate_hz: fixture.sample_rate_hz,
        channels: fixture.channels,
        samples: fixture.samples.clone(),
    });

    let _ = measure_loopback_latency_ms(&transport).await;
    let sent = transport.sent_frames();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].sample_rate_hz, fixture.sample_rate_hz);
    assert_eq!(sent[0].channels, fixture.channels);
    assert_eq!(sent[0].samples, fixture.samples);
}

#[tokio::test]
async fn scripted_transport_loopback_latency_stays_low_for_fixture_audio() {
    let fixture = load_waveform_fixture("impulse-16khz-mono.json");
    let transport = ScriptedTransport::default();
    transport.push_input(AudioInputFrame {
        sequence: 7,
        sample_rate_hz: fixture.sample_rate_hz,
        channels: fixture.channels,
        samples: fixture.samples,
    });

    let elapsed_ms = measure_loopback_latency_ms(&transport).await;
    assert!(
        elapsed_ms < 50,
        "expected in-process loopback under 50ms, got {elapsed_ms}ms"
    );
}
