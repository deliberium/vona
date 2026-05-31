use crate::types::{AudioInputFrame, AudioOutputFrame};
use async_trait::async_trait;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextGenerationInput {
    pub prompt: String,
    pub stream: bool,
}

impl TextGenerationInput {
    pub fn streaming(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            stream: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextGenerationFrame {
    pub text: String,
    pub final_fragment: bool,
}

#[derive(Debug, Error)]
pub enum TextGenerationError {
    #[error("text generation start failed: {0}")]
    Start(String),
    #[error("text generation stream failed: {0}")]
    Stream(String),
}

pub type TokenStream =
    Pin<Box<dyn Stream<Item = Result<TextGenerationFrame, TextGenerationError>> + Send + 'static>>;

pub trait TextGenerator: Send + Sync {
    fn generate_text(&self, input: TextGenerationInput) -> TokenStream;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioSynthesisConfig {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

#[derive(Debug, Error)]
pub enum AudioProcessingError {
    #[error("audio runtime failed: {0}")]
    Runtime(String),
    #[error("audio model is unavailable: {0}")]
    ModelUnavailable(String),
    #[error("audio input is invalid: {0}")]
    InvalidInput(String),
    #[error("audio inference failed: {0}")]
    Inference(String),
}

#[async_trait]
pub trait AudioTranscriber: Send + Sync {
    async fn transcribe_audio(
        &self,
        input: AudioInputFrame,
    ) -> Result<String, AudioProcessingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingTranscriptionConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub step_ms: u32,
    pub min_buffer_ms: u32,
    pub max_buffer_ms: u32,
    pub stability_passes: u32,
}

impl StreamingTranscriptionConfig {
    pub fn new(sample_rate_hz: u32, channels: u16) -> Self {
        Self {
            sample_rate_hz,
            channels,
            step_ms: 600,
            min_buffer_ms: 1_200,
            max_buffer_ms: 30_000,
            stability_passes: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingTranscriptKind {
    Partial,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamingTranscriptUpdate {
    pub kind: StreamingTranscriptKind,
    pub text: String,
    pub stability_passes: u32,
    pub total_audio_ms: u64,
}

#[async_trait]
pub trait StreamingTranscriptionSession: Send {
    async fn push_audio(
        &mut self,
        input: AudioInputFrame,
    ) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError>;

    async fn finish(&mut self) -> Result<Option<StreamingTranscriptUpdate>, AudioProcessingError>;
}

#[async_trait]
pub trait AudioStreamingTranscriber: Send + Sync {
    async fn start_streaming_transcription(
        &self,
        config: StreamingTranscriptionConfig,
    ) -> Result<Box<dyn StreamingTranscriptionSession>, AudioProcessingError>;
}

#[async_trait]
pub trait AudioSynthesizer: Send + Sync {
    async fn synthesize_audio(
        &self,
        text: String,
        config: AudioSynthesisConfig,
    ) -> Result<AudioOutputFrame, AudioProcessingError>;
}
