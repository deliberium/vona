use crate::session::SessionConfig;
use crate::types::{AudioInputFrame, AudioOutputFrame, ControlEvent, ExternalContextEvent};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_full_duplex: bool,
    pub supports_control_stream: bool,
    pub supports_context_injection: bool,
    pub supports_pause_resume: bool,
    pub supports_style_conditioning: bool,
    pub supports_word_timestamps: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_full_duplex: true,
            supports_control_stream: true,
            supports_context_injection: false,
            supports_pause_resume: true,
            supports_style_conditioning: false,
            supports_word_timestamps: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendStep {
    pub output_audio: Vec<AudioOutputFrame>,
    pub control_events: Vec<ControlEvent>,
    pub transcript: Option<String>,
    pub finished: bool,
    pub debug_payload: Option<Value>,
}

impl Default for BackendStep {
    fn default() -> Self {
        Self {
            output_audio: Vec::new(),
            control_events: Vec::new(),
            transcript: None,
            finished: false,
            debug_payload: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend session start failed: {0}")]
    Start(String),
    #[error("backend step failed: {0}")]
    Step(String),
    #[error("backend context injection failed: {0}")]
    Inject(String),
    #[error("backend end session failed: {0}")]
    End(String),
}

#[async_trait]
pub trait SpeechToSpeechBackend: Send + Sync {
    type Session: Send + Sync;

    fn capabilities(&self) -> BackendCapabilities;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError>;

    async fn step(
        &self,
        session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError>;

    async fn inject_event(
        &self,
        session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError>;

    async fn end_session(&self, session: Self::Session) -> Result<(), BackendError>;
}
