use crate::remote::{
    SeamlessM4tRemoteBackend, SeamlessM4tRemoteConfig, SeamlessM4tRemoteSession,
    SeamlessM4tRemoteStepRequest, SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransport,
    SeamlessM4tRemoteTransportError,
};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;
use vona::{
    BackendCapabilities, BackendError, BackendStep, ExternalContextEvent, SessionConfig,
    SpeechToSpeechBackend,
};

#[derive(Debug, Clone)]
pub struct SeamlessM4tHttpConfig {
    pub base_url: String,
    pub model: Option<String>,
    pub bearer_token: Option<String>,
    pub timeout_ms: u64,
}

impl SeamlessM4tHttpConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: None,
            bearer_token: None,
            timeout_ms: 15_000,
        }
    }
}

#[derive(Clone)]
struct SeamlessM4tHttpTransport {
    client: Client,
    config: SeamlessM4tHttpConfig,
}

impl SeamlessM4tHttpTransport {
    fn new(config: SeamlessM4tHttpConfig) -> Result<Self, BackendError> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|err| BackendError::Start(format!("failed to build HTTP client: {err}")))?;

        Ok(Self { client, config })
    }

    fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            path.trim_start_matches('/'),
        )
    }
}

#[async_trait]
impl SeamlessM4tRemoteTransport for SeamlessM4tHttpTransport {
    async fn step(
        &self,
        request: SeamlessM4tRemoteStepRequest,
    ) -> Result<SeamlessM4tRemoteStepResponse, SeamlessM4tRemoteTransportError> {
        let mut http_request = self.client.post(self.endpoint("v1/seamless-m4t/step"));
        if let Some(token) = &self.config.bearer_token {
            http_request = http_request.bearer_auth(token);
        }

        let response = http_request
            .json(&request)
            .send()
            .await
            .map_err(|err| SeamlessM4tRemoteTransportError::Request(err.to_string()))?;

        let response = if response.status().is_success() {
            response
        } else {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|err| format!("<failed to read error body: {err}>"));
            let detail = body.trim();
            let message = if detail.is_empty() {
                format!("backend returned error status: {status}")
            } else {
                format!("backend returned error status: {status}: {detail}")
            };
            return Err(SeamlessM4tRemoteTransportError::Response(message));
        };

        response.json().await.map_err(|err| {
            SeamlessM4tRemoteTransportError::Response(format!("invalid backend response: {err}"))
        })
    }
}

#[derive(Clone)]
pub struct SeamlessM4tHttpBackend {
    inner: SeamlessM4tRemoteBackend<SeamlessM4tHttpTransport>,
}

pub type SeamlessM4tHttpSession = SeamlessM4tRemoteSession;

impl SeamlessM4tHttpBackend {
    pub fn new(config: SeamlessM4tHttpConfig) -> Result<Self, BackendError> {
        let transport = SeamlessM4tHttpTransport::new(config.clone())?;
        Ok(Self {
            inner: SeamlessM4tRemoteBackend::new(
                transport,
                SeamlessM4tRemoteConfig {
                    model: config.model.clone(),
                },
            ),
        })
    }
}

#[async_trait]
impl SpeechToSpeechBackend for SeamlessM4tHttpBackend {
    type Session = SeamlessM4tHttpSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.inner.capabilities()
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        self.inner.start_session(config).await
    }

    async fn step(
        &self,
        session: &mut Self::Session,
        input: vona::AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        self.inner.step(session, input).await
    }

    async fn inject_event(
        &self,
        session: &mut Self::Session,
        event: ExternalContextEvent,
    ) -> Result<(), BackendError> {
        self.inner.inject_event(session, event).await
    }

    async fn end_session(&self, session: Self::Session) -> Result<(), BackendError> {
        self.inner.end_session(session).await
    }
}
