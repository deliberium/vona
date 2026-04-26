use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_cbor::{from_slice, to_vec};
use std::io::{ErrorKind, Result as IoResult};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Mutex;
use vona_seamless::{
    SeamlessM4tRemoteBackend, SeamlessM4tRemoteConfig, SeamlessM4tRemoteStepRequest,
    SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransport, SeamlessM4tRemoteTransportError,
};

/// Maximum allowed CBOR frame size (4 MiB). Frames larger than this are rejected
/// to prevent memory exhaustion from malformed or hostile senders.
const MAX_FRAME_BYTES: u32 = 4 * 1024 * 1024;

/// Error produced when an incoming CBOR frame exceeds [`MAX_FRAME_BYTES`].
#[derive(Debug, Error)]
#[error("frame too large: {bytes} bytes exceeds {limit} byte limit")]
pub struct FrameTooLarge {
    pub bytes: u32,
    pub limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalIpcStepEnvelope {
    pub response: Option<SeamlessM4tRemoteStepResponse>,
    pub error: Option<String>,
}

pub async fn write_length_prefixed_frame<W, T>(writer: &mut W, value: &T) -> IoResult<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let payload = to_vec(value)
        .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
    writer.write_u32_le(payload.len() as u32).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_length_prefixed_frame<R, T>(reader: &mut R) -> IoResult<Option<T>>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let frame_len = match reader.read_u32_le().await {
        Ok(len) => len,
        Err(err) if err.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    };

    if frame_len > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            FrameTooLarge { bytes: frame_len, limit: MAX_FRAME_BYTES },
        ));
    }

    let mut payload = vec![0_u8; frame_len as usize];
    reader.read_exact(&mut payload).await?;
    let decoded = from_slice(&payload)
        .map_err(|err| std::io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
    Ok(Some(decoded))
}

#[derive(Debug, Clone)]
pub struct LocalIpcTransportConfig {
    pub endpoint: PathBuf,
    pub connect_timeout_ms: u64,
}

impl LocalIpcTransportConfig {
    pub fn unix_socket(endpoint: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: endpoint.into(),
            connect_timeout_ms: 5_000,
        }
    }
}

#[derive(Debug, Error)]
pub enum LocalIpcTransportInitError {
    #[error("local IPC transport is unsupported on this platform")]
    UnsupportedPlatform,
}

#[cfg(unix)]
type StreamHandle = tokio::net::UnixStream;

#[cfg(not(unix))]
struct StreamHandle;

#[derive(Clone)]
pub struct LocalIpcSeamlessM4tTransport {
    config: LocalIpcTransportConfig,
    stream: Arc<Mutex<Option<StreamHandle>>>,
}

impl LocalIpcSeamlessM4tTransport {
    pub fn new(config: LocalIpcTransportConfig) -> Result<Self, LocalIpcTransportInitError> {
        #[cfg(not(unix))]
        {
            let _ = config;
            return Err(LocalIpcTransportInitError::UnsupportedPlatform);
        }

        #[cfg(unix)]
        {
            Ok(Self {
                config,
                stream: Arc::new(Mutex::new(None)),
            })
        }
    }

    pub fn backend(
        config: LocalIpcTransportConfig,
        model: Option<String>,
    ) -> Result<LocalIpcSeamlessM4tBackend, LocalIpcTransportInitError> {
        let transport = Self::new(config)?;
        Ok(SeamlessM4tRemoteBackend::new(
            transport,
            SeamlessM4tRemoteConfig { model },
        ))
    }

    #[cfg(unix)]
    async fn connect(&self) -> Result<StreamHandle, SeamlessM4tRemoteTransportError> {
        tokio::time::timeout(
            tokio::time::Duration::from_millis(self.config.connect_timeout_ms),
            tokio::net::UnixStream::connect(&self.config.endpoint),
        )
        .await
        .map_err(|_| {
            SeamlessM4tRemoteTransportError::Request(format!(
                "timed out connecting to local IPC endpoint {}",
                self.config.endpoint.display()
            ))
        })?
        .map_err(|err| {
            SeamlessM4tRemoteTransportError::Request(format!(
                "failed to connect to local IPC endpoint {}: {err}",
                self.config.endpoint.display()
            ))
        })
    }
}

pub type LocalIpcSeamlessM4tBackend = SeamlessM4tRemoteBackend<LocalIpcSeamlessM4tTransport>;

#[async_trait]
impl SeamlessM4tRemoteTransport for LocalIpcSeamlessM4tTransport {
    async fn step(
        &self,
        request: SeamlessM4tRemoteStepRequest,
    ) -> Result<SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransportError> {
        #[cfg(not(unix))]
        {
            let _ = request;
            return Err(SeamlessM4tRemoteTransportError::Request(
                "local IPC transport is unsupported on this platform".to_string(),
            ));
        }

        #[cfg(unix)]
        {
            let mut guard = self.stream.lock().await;
            if guard.is_none() {
                *guard = Some(self.connect().await?);
            }

            let result = async {
                let stream = guard.as_mut().expect("stream should be present");
                write_length_prefixed_frame(stream, &request)
                    .await
                    .map_err(|err| SeamlessM4tRemoteTransportError::Request(err.to_string()))?;
                let envelope: LocalIpcStepEnvelope = read_length_prefixed_frame(stream)
                    .await
                    .map_err(|err| SeamlessM4tRemoteTransportError::Response(err.to_string()))?
                    .ok_or_else(|| {
                        SeamlessM4tRemoteTransportError::Response(
                            "local IPC transport closed before delivering a response".to_string(),
                        )
                    })?;
                if let Some(error) = envelope.error {
                    Err(SeamlessM4tRemoteTransportError::Response(error))
                } else {
                    envelope.response.ok_or_else(|| {
                        SeamlessM4tRemoteTransportError::Response(
                            "local IPC response did not include a backend payload".to_string(),
                        )
                    })
                }
            }
            .await;

            if result.is_err() {
                *guard = None;
            }

            result
        }
    }
}
