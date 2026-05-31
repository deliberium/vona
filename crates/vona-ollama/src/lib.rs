use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;
use tokio::sync::mpsc;
use vona_core::{
    RealtimeVoiceBackend, RealtimeVoiceCapabilities, RealtimeVoiceControl, RealtimeVoiceError,
    RealtimeVoiceInput, RealtimeVoiceModelFamily, RealtimeVoiceOutput, RealtimeVoiceSessionConfig,
    TextGenerationError, TextGenerationFrame, TextGenerationInput, TextGenerator, TokenStream,
};

pub const DEFAULT_API_BASE: &str = "http://localhost:11434";
pub const DEFAULT_MODEL: &str = "phi4-mini";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub api_base: String,
    pub model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            model: DEFAULT_MODEL.to_string(),
        }
    }
}

impl OllamaConfig {
    pub fn new(model: Option<String>) -> Self {
        Self {
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            ..Self::default()
        }
    }

    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
        }
    }

    pub fn generate_url(&self) -> String {
        format!("{}/api/generate", self.api_base.trim_end_matches('/'))
    }

    pub fn session_config(&self, session_id: impl Into<String>) -> RealtimeVoiceSessionConfig {
        RealtimeVoiceSessionConfig {
            session_id: session_id.into(),
            input_sample_rate_hz: 0,
            output_sample_rate_hz: 0,
            channels: 0,
            model_family: RealtimeVoiceModelFamily::HostedRealtimeApi {
                provider: "ollama".to_string(),
                model: self.model.clone(),
            },
            metadata: json!({ "api_base": self.api_base }),
        }
    }
}

#[derive(Clone)]
pub struct OllamaTextEngine {
    client: reqwest::Client,
    config: OllamaConfig,
}

impl Default for OllamaTextEngine {
    fn default() -> Self {
        Self::new(OllamaConfig::default())
    }
}

impl OllamaTextEngine {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    pub fn with_model(model: Option<String>) -> Self {
        Self::new(OllamaConfig::new(model))
    }

    pub fn from_env() -> Self {
        Self::new(OllamaConfig::from_env())
    }

    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }
}

impl TextGenerator for OllamaTextEngine {
    fn generate_text(&self, input: TextGenerationInput) -> TokenStream {
        let rx = spawn_text_generation_stream(
            self.client.clone(),
            self.config.clone(),
            input.prompt,
            input.stream,
        );
        Box::pin(stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }))
    }
}

#[derive(Debug)]
pub struct OllamaSession {
    pub config: RealtimeVoiceSessionConfig,
    pending_prompt: Option<String>,
    rx: Option<mpsc::Receiver<Result<RealtimeVoiceOutput, OllamaError>>>,
}

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("Ollama request failed: {0}")]
    Request(String),
    #[error("Ollama response stream failed: {0}")]
    Stream(String),
    #[error("Ollama returned malformed JSON: {0}")]
    Json(String),
}

impl From<OllamaError> for RealtimeVoiceError {
    fn from(value: OllamaError) -> Self {
        RealtimeVoiceError::Receive(value.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateChunk {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
}

#[async_trait]
impl RealtimeVoiceBackend for OllamaTextEngine {
    type Session = OllamaSession;

    fn realtime_capabilities(&self) -> RealtimeVoiceCapabilities {
        RealtimeVoiceCapabilities {
            supports_full_duplex: false,
            supports_streaming_audio_input: false,
            supports_streaming_audio_output: false,
            supports_tool_calls: false,
            supports_interruption: false,
            supports_context_injection: false,
            is_hosted_service: false,
            max_input_chunk_ms: None,
        }
    }

    async fn start_realtime_session(
        &self,
        config: RealtimeVoiceSessionConfig,
    ) -> Result<Self::Session, RealtimeVoiceError> {
        Ok(OllamaSession {
            config,
            pending_prompt: None,
            rx: None,
        })
    }

    async fn send_realtime_event(
        &self,
        session: &mut Self::Session,
        input: RealtimeVoiceInput,
    ) -> Result<(), RealtimeVoiceError> {
        match input {
            RealtimeVoiceInput::Text { text } => {
                session.pending_prompt = Some(text.clone());
                session.rx = Some(spawn_generate_stream(
                    self.client.clone(),
                    self.config.clone(),
                    text,
                ));
                Ok(())
            }
            RealtimeVoiceInput::Control(RealtimeVoiceControl::StartResponse) => {
                let Some(prompt) = session.pending_prompt.clone() else {
                    return Err(RealtimeVoiceError::Send(
                        "cannot start Ollama response without prior text input".to_string(),
                    ));
                };
                session.rx = Some(spawn_generate_stream(
                    self.client.clone(),
                    self.config.clone(),
                    prompt,
                ));
                Ok(())
            }
            RealtimeVoiceInput::Control(RealtimeVoiceControl::Close) => {
                session.rx = None;
                Ok(())
            }
            RealtimeVoiceInput::Audio(_) => Err(RealtimeVoiceError::Send(
                "vona-ollama accepts text input only".to_string(),
            )),
            RealtimeVoiceInput::ToolResult(_) | RealtimeVoiceInput::Control(_) => Ok(()),
        }
    }

    async fn recv_realtime_event(
        &self,
        session: &mut Self::Session,
    ) -> Result<Option<RealtimeVoiceOutput>, RealtimeVoiceError> {
        let Some(rx) = &mut session.rx else {
            return Ok(None);
        };

        match rx.recv().await {
            Some(Ok(output)) => Ok(Some(output)),
            Some(Err(error)) => Err(error.into()),
            None => {
                session.rx = None;
                Ok(None)
            }
        }
    }

    async fn close_realtime_session(
        &self,
        mut session: Self::Session,
    ) -> Result<(), RealtimeVoiceError> {
        session.rx = None;
        Ok(())
    }
}

fn spawn_generate_stream(
    client: reqwest::Client,
    config: OllamaConfig,
    prompt: String,
) -> mpsc::Receiver<Result<RealtimeVoiceOutput, OllamaError>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        if let Err(error) = stream_generate(client, config, prompt, tx.clone()).await {
            let _ = tx.send(Err(error)).await;
        }
    });
    rx
}

fn spawn_text_generation_stream(
    client: reqwest::Client,
    config: OllamaConfig,
    prompt: String,
    stream_response: bool,
) -> mpsc::Receiver<Result<TextGenerationFrame, TextGenerationError>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        if let Err(error) =
            stream_text_generation(client, config, prompt, stream_response, tx.clone()).await
        {
            let _ = tx.send(Err(error)).await;
        }
    });
    rx
}

async fn stream_generate(
    client: reqwest::Client,
    config: OllamaConfig,
    prompt: String,
    tx: mpsc::Sender<Result<RealtimeVoiceOutput, OllamaError>>,
) -> Result<(), OllamaError> {
    let response = client
        .post(config.generate_url())
        .json(&json!({
            "model": config.model,
            "prompt": prompt,
            "stream": true,
        }))
        .send()
        .await
        .map_err(|error| OllamaError::Request(error.to_string()))?
        .error_for_status()
        .map_err(|error| OllamaError::Request(error.to_string()))?;

    let mut buffered = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| OllamaError::Stream(error.to_string()))?;
        buffered.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline) = buffered.find('\n') {
            let line = buffered[..newline].trim().to_string();
            buffered = buffered[newline + 1..].to_string();
            if !line.is_empty() {
                handle_generate_line(&line, &tx).await?;
            }
        }
    }

    if !buffered.trim().is_empty() {
        handle_generate_line(buffered.trim(), &tx).await?;
    }

    Ok(())
}

async fn stream_text_generation(
    client: reqwest::Client,
    config: OllamaConfig,
    prompt: String,
    stream_response: bool,
    tx: mpsc::Sender<Result<TextGenerationFrame, TextGenerationError>>,
) -> Result<(), TextGenerationError> {
    let response = client
        .post(config.generate_url())
        .json(&json!({
            "model": config.model,
            "prompt": prompt,
            "stream": stream_response,
        }))
        .send()
        .await
        .map_err(|error| TextGenerationError::Start(error.to_string()))?
        .error_for_status()
        .map_err(|error| TextGenerationError::Start(error.to_string()))?;

    let mut buffered = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| TextGenerationError::Stream(error.to_string()))?;
        buffered.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline) = buffered.find('\n') {
            let line = buffered[..newline].trim().to_string();
            buffered = buffered[newline + 1..].to_string();
            if !line.is_empty() {
                handle_text_generation_line(&line, &tx).await?;
            }
        }
    }

    if !buffered.trim().is_empty() {
        handle_text_generation_line(buffered.trim(), &tx).await?;
    }

    Ok(())
}

async fn handle_generate_line(
    line: &str,
    tx: &mpsc::Sender<Result<RealtimeVoiceOutput, OllamaError>>,
) -> Result<(), OllamaError> {
    let chunk: OllamaGenerateChunk =
        serde_json::from_str(line).map_err(|error| OllamaError::Json(error.to_string()))?;

    if let Some(error) = chunk.error {
        return Err(OllamaError::Stream(error));
    }

    if !chunk.response.is_empty() {
        let _ = tx
            .send(Ok(RealtimeVoiceOutput::TranscriptFragment {
                text: chunk.response,
                final_fragment: false,
            }))
            .await;
    }

    if chunk.done {
        let _ = tx
            .send(Ok(RealtimeVoiceOutput::ResponseCompleted {
                reason: Some("ollama.done".to_string()),
            }))
            .await;
    }

    Ok(())
}

async fn handle_text_generation_line(
    line: &str,
    tx: &mpsc::Sender<Result<TextGenerationFrame, TextGenerationError>>,
) -> Result<(), TextGenerationError> {
    let chunk: OllamaGenerateChunk = serde_json::from_str(line)
        .map_err(|error| TextGenerationError::Stream(error.to_string()))?;

    if let Some(error) = chunk.error {
        return Err(TextGenerationError::Stream(error));
    }

    if !chunk.response.is_empty() || chunk.done {
        let _ = tx
            .send(Ok(TextGenerationFrame {
                text: chunk.response,
                final_fragment: chunk.done,
            }))
            .await;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builds_generate_url() {
        let config = OllamaConfig {
            api_base: "http://localhost:11434/".to_string(),
            model: "phi4-mini".to_string(),
        };
        assert_eq!(config.generate_url(), "http://localhost:11434/api/generate");
    }

    #[tokio::test]
    async fn parses_response_lines() {
        let (tx, mut rx) = mpsc::channel(4);
        handle_generate_line(r#"{"response":"hello","done":false}"#, &tx)
            .await
            .unwrap();
        handle_generate_line(r#"{"response":"","done":true}"#, &tx)
            .await
            .unwrap();

        assert_eq!(
            rx.recv().await.unwrap().unwrap(),
            RealtimeVoiceOutput::TranscriptFragment {
                text: "hello".to_string(),
                final_fragment: false,
            }
        );
        assert_eq!(
            rx.recv().await.unwrap().unwrap(),
            RealtimeVoiceOutput::ResponseCompleted {
                reason: Some("ollama.done".to_string()),
            }
        );
    }
}
