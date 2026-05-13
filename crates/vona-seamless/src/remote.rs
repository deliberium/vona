use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use vona_core::{
    AudioInputFrame, AudioOutputFrame, BackendCapabilities, BackendError, BackendStep,
    ControlEvent, ExternalContextEvent, SessionConfig, SpeechStyleProfile, SpeechToSpeechBackend,
};

#[derive(Debug, Clone, Default)]
pub struct SeamlessM4tRemoteConfig {
    pub model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SeamlessM4tRemoteSession {
    pub config: SessionConfig,
    pub pending_events: Vec<ExternalContextEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeamlessM4tRemoteStepRequest {
    pub session_id: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub input_samples: Vec<f32>,
    pub model: Option<String>,
    pub session_metadata: Value,
    pub style_profile: Option<SpeechStyleProfile>,
    pub pending_events: Vec<ExternalContextEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeamlessM4tRemoteStepResponse {
    #[serde(default)]
    pub output_samples: Vec<f32>,
    #[serde(default = "default_output_sample_rate_hz")]
    pub output_sample_rate_hz: u32,
    #[serde(default)]
    pub transcript: Option<String>,
    #[serde(default)]
    pub control_events: Vec<ControlEvent>,
    #[serde(default)]
    pub finished: bool,
    #[serde(default)]
    pub debug_payload: Option<Value>,
}

fn default_output_sample_rate_hz() -> u32 {
    16_000
}

#[derive(Debug, Error)]
pub enum SeamlessM4tRemoteTransportError {
    #[error("remote request failed: {0}")]
    Request(String),
    #[error("remote response failed: {0}")]
    Response(String),
}

#[async_trait]
pub trait SeamlessM4tRemoteTransport: Send + Sync {
    async fn step(
        &self,
        request: SeamlessM4tRemoteStepRequest,
    ) -> Result<SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransportError>;
}

#[derive(Clone)]
pub struct SeamlessM4tRemoteBackend<T> {
    transport: T,
    config: SeamlessM4tRemoteConfig,
}

impl<T> SeamlessM4tRemoteBackend<T> {
    pub fn new(transport: T, config: SeamlessM4tRemoteConfig) -> Self {
        Self { transport, config }
    }
}

#[async_trait]
impl<T> SpeechToSpeechBackend for SeamlessM4tRemoteBackend<T>
where
    T: SeamlessM4tRemoteTransport,
{
    type Session = SeamlessM4tRemoteSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_context_injection: true,
            supports_style_conditioning: true,
            ..BackendCapabilities::default()
        }
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(SeamlessM4tRemoteSession {
            config,
            pending_events: Vec::new(),
        })
    }

    async fn step(
        &self,
        session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        let payload = self
            .transport
            .step(SeamlessM4tRemoteStepRequest {
                session_id: session.config.session_id.clone(),
                sample_rate_hz: input.sample_rate_hz,
                channels: input.channels,
                input_samples: input.samples,
                model: self.config.model.clone(),
                session_metadata: session.config.metadata.clone(),
                style_profile: session.config.style_profile.clone(),
                pending_events: std::mem::take(&mut session.pending_events),
            })
            .await
            .map_err(|err| BackendError::Step(err.to_string()))?;

        Ok(BackendStep {
            output_audio: vec![AudioOutputFrame {
                sequence: input.sequence,
                sample_rate_hz: payload.output_sample_rate_hz,
                // Use the session config channel count instead of hardcoding 1
                channels: session.config.channels,
                samples: payload.output_samples,
                is_filler: false,
            }],
            control_events: payload.control_events,
            transcript: payload.transcript,
            finished: payload.finished,
            debug_payload: payload.debug_payload,
        })
    }

    async fn inject_event(
        &self,
        session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        session.pending_events.push(event);
        Ok(())
    }

    async fn end_session(&self, _session: Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vona_core::{AudioInputFrame, SpeechToSpeechBackend};

    // ── Mock transport ──────────────────────────────────────────────────────

    /// Returns a fixed response for every step call.
    struct EchoTransport {
        response: SeamlessM4tRemoteStepResponse,
    }

    #[async_trait::async_trait]
    impl SeamlessM4tRemoteTransport for EchoTransport {
        async fn step(
            &self,
            _request: SeamlessM4tRemoteStepRequest,
        ) -> Result<SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransportError> {
            Ok(self.response.clone())
        }
    }

    fn echo_transport(samples: Vec<f32>) -> EchoTransport {
        EchoTransport {
            response: SeamlessM4tRemoteStepResponse {
                output_samples: samples,
                output_sample_rate_hz: 16_000,
                transcript: Some("test transcript".into()),
                control_events: vec![],
                finished: false,
                debug_payload: None,
            },
        }
    }

    fn input_frame(samples: Vec<f32>) -> AudioInputFrame {
        AudioInputFrame {
            sequence: 1,
            sample_rate_hz: 16_000,
            channels: 1,
            samples,
        }
    }

    // ── SeamlessM4tRemoteConfig ─────────────────────────────────────────────

    #[test]
    fn remote_config_default_has_no_model() {
        let cfg = SeamlessM4tRemoteConfig::default();
        assert!(cfg.model.is_none());
    }

    // ── BackendCapabilities ─────────────────────────────────────────────────

    #[test]
    fn remote_backend_capabilities() {
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![]),
            SeamlessM4tRemoteConfig::default(),
        );
        let caps = backend.capabilities();
        assert!(caps.supports_context_injection);
        assert!(caps.supports_style_conditioning);
    }

    // ── start_session ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn remote_backend_start_session() {
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![]),
            SeamlessM4tRemoteConfig::default(),
        );
        let session_cfg = vona_core::SessionConfig {
            session_id: "test-session".into(),
            sample_rate_hz: 16_000,
            channels: 1,
            ..Default::default()
        };
        let session = backend.start_session(session_cfg.clone()).await.unwrap();
        assert_eq!(session.config.session_id, "test-session");
        assert!(session.pending_events.is_empty());
    }

    // ── step ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn remote_backend_step_returns_echo_samples() {
        let echo_samples: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(echo_samples.clone()),
            SeamlessM4tRemoteConfig::default(),
        );
        let mut session = backend
            .start_session(vona_core::SessionConfig::default())
            .await
            .unwrap();

        let result = backend
            .step(&mut session, input_frame(vec![0.1, 0.2, 0.3]))
            .await
            .unwrap();

        assert_eq!(result.output_audio.len(), 1);
        assert_eq!(result.output_audio[0].samples, echo_samples);
        assert_eq!(result.output_audio[0].sample_rate_hz, 16_000);
        assert_eq!(result.transcript.as_deref(), Some("test transcript"));
        assert!(!result.finished);
    }

    #[tokio::test]
    async fn remote_backend_step_preserves_sequence() {
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![1.0]),
            SeamlessM4tRemoteConfig::default(),
        );
        let mut session = backend
            .start_session(vona_core::SessionConfig::default())
            .await
            .unwrap();

        let frame = AudioInputFrame {
            sequence: 42,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0],
        };
        let result = backend.step(&mut session, frame).await.unwrap();
        assert_eq!(result.output_audio[0].sequence, 42);
    }

    // ── inject_event ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn remote_backend_inject_event_queues_it() {
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![]),
            SeamlessM4tRemoteConfig::default(),
        );
        let mut session = backend
            .start_session(vona_core::SessionConfig::default())
            .await
            .unwrap();

        let event = ExternalContextEvent {
            source: "vona.transcript_override".into(),
            spoken_summary: None,
            payload: serde_json::json!("override text"),
        };
        backend
            .inject_event(&mut session, event.clone())
            .await
            .unwrap();
        assert_eq!(session.pending_events.len(), 1);
        assert_eq!(session.pending_events[0].source, "vona.transcript_override");
    }

    #[tokio::test]
    async fn remote_backend_step_drains_pending_events() {
        // Verify pending events are consumed (moved into the request) on step.
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![]),
            SeamlessM4tRemoteConfig::default(),
        );
        let mut session = backend
            .start_session(vona_core::SessionConfig::default())
            .await
            .unwrap();

        backend
            .inject_event(
                &mut session,
                ExternalContextEvent {
                    source: "vona.plan_result".into(),
                    spoken_summary: Some("plan text".into()),
                    payload: serde_json::json!(null),
                },
            )
            .await
            .unwrap();

        assert_eq!(session.pending_events.len(), 1);
        let _ = backend
            .step(&mut session, input_frame(vec![0.0]))
            .await
            .unwrap();
        assert!(
            session.pending_events.is_empty(),
            "pending events should be drained after step"
        );
    }

    // ── end_session ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn remote_backend_end_session_is_ok() {
        let backend = SeamlessM4tRemoteBackend::new(
            echo_transport(vec![]),
            SeamlessM4tRemoteConfig::default(),
        );
        let session = backend
            .start_session(vona_core::SessionConfig::default())
            .await
            .unwrap();
        assert!(backend.end_session(session).await.is_ok());
    }
}
