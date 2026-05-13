use crate::types::{AudioInputFrame, AudioOutputFrame, ExternalContextEvent, SkillCall};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeVoiceModelFamily {
    HostedRealtimeApi { provider: String, model: String },
    MoshiFamily { variant: String },
    SeamlessFamily { variant: String },
    OpenRealtimeModel { name: String },
    Custom { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealtimeVoiceSessionConfig {
    pub session_id: String,
    pub input_sample_rate_hz: u32,
    pub output_sample_rate_hz: u32,
    pub channels: u16,
    pub model_family: RealtimeVoiceModelFamily,
    #[serde(default)]
    pub metadata: Value,
}

impl Default for RealtimeVoiceSessionConfig {
    fn default() -> Self {
        Self {
            session_id: "default-realtime-session".to_string(),
            input_sample_rate_hz: 24_000,
            output_sample_rate_hz: 24_000,
            channels: 1,
            model_family: RealtimeVoiceModelFamily::Custom {
                name: "unspecified".to_string(),
            },
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeVoiceCapabilities {
    pub supports_full_duplex: bool,
    pub supports_streaming_audio_input: bool,
    pub supports_streaming_audio_output: bool,
    pub supports_tool_calls: bool,
    pub supports_interruption: bool,
    pub supports_context_injection: bool,
    pub is_hosted_service: bool,
    pub max_input_chunk_ms: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeVoiceControl {
    StartResponse,
    CommitInput,
    ClearOutput,
    Interrupt { reason: Option<String> },
    Close,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeVoiceInput {
    Audio(AudioInputFrame),
    Text { text: String },
    ToolResult(ExternalContextEvent),
    Control(RealtimeVoiceControl),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeLatencyStage {
    InputReceived,
    FirstAudio,
    ToolCallEmitted,
    OutputCleared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeLatencyMark {
    pub stage: RealtimeLatencyStage,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeVoiceOutput {
    Audio(AudioOutputFrame),
    TranscriptFragment { text: String, final_fragment: bool },
    ToolCall(SkillCall),
    Interruption { reason: Option<String> },
    LatencyMark(RealtimeLatencyMark),
    Closed { reason: Option<String> },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RealtimeVoiceError {
    #[error("realtime session start failed: {0}")]
    Start(String),
    #[error("realtime send failed: {0}")]
    Send(String),
    #[error("realtime receive failed: {0}")]
    Receive(String),
    #[error("realtime session close failed: {0}")]
    Close(String),
}

#[async_trait]
pub trait RealtimeVoiceBackend: Send + Sync {
    type Session: Send + Sync;

    fn realtime_capabilities(&self) -> RealtimeVoiceCapabilities;

    async fn start_realtime_session(
        &self,
        config: RealtimeVoiceSessionConfig,
    ) -> Result<Self::Session, RealtimeVoiceError>;

    async fn send_realtime_event(
        &self,
        session: &mut Self::Session,
        input: RealtimeVoiceInput,
    ) -> Result<(), RealtimeVoiceError>;

    async fn recv_realtime_event(
        &self,
        session: &mut Self::Session,
    ) -> Result<Option<RealtimeVoiceOutput>, RealtimeVoiceError>;

    async fn close_realtime_session(
        &self,
        session: Self::Session,
    ) -> Result<(), RealtimeVoiceError>;
}
