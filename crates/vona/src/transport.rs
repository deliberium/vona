use crate::types::{AudioInputFrame, AudioOutputFrame};
use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("transport receive failed: {0}")]
    Receive(String),
    #[error("transport send failed: {0}")]
    Send(String),
    #[error("transport clear failed: {0}")]
    Clear(String),
}

#[async_trait]
pub trait AudioTransport: Send + Sync {
    fn sample_rate_hz(&self) -> u32;
    fn channels(&self) -> u16;
    async fn recv_frame(&self) -> Result<Option<AudioInputFrame>, TransportError>;
    async fn send_frame(&self, frame: AudioOutputFrame) -> Result<(), TransportError>;
    async fn clear_output(&self) -> Result<(), TransportError>;
}
