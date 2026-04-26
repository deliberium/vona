use crate::backend::{BackendCapabilities, BackendError, BackendStep, SpeechToSpeechBackend};
use crate::session::SessionConfig;
use crate::types::{AudioInputFrame, AudioOutputFrame, ExternalContextEvent};
use async_trait::async_trait;

#[derive(Debug, Clone, Default)]
pub struct PassthroughStsBackend;

#[derive(Debug, Clone)]
pub struct PassthroughSession {
    pub config: SessionConfig,
    pub output_sequence: u64,
    pub injected_events: Vec<ExternalContextEvent>,
}

#[async_trait]
impl SpeechToSpeechBackend for PassthroughStsBackend {
    type Session = PassthroughSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_context_injection: true,
            supports_style_conditioning: true,
            ..BackendCapabilities::default()
        }
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(PassthroughSession {
            config,
            output_sequence: 0,
            injected_events: Vec::new(),
        })
    }

    async fn step(
        &self,
        session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        session.output_sequence = session.output_sequence.saturating_add(1);
        let frame = AudioOutputFrame {
            sequence: session.output_sequence,
            sample_rate_hz: input.sample_rate_hz,
            channels: input.channels,
            samples: input.samples,
            is_filler: false,
        };

        Ok(BackendStep {
            output_audio: vec![frame],
            transcript: Some("passthrough_audio".to_string()),
            ..BackendStep::default()
        })
    }

    async fn inject_event(
        &self,
        session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        session.injected_events.push(event);
        Ok(())
    }

    async fn end_session(&self, _session: Self::Session) -> Result<(), BackendError> {
        Ok(())
    }
}
