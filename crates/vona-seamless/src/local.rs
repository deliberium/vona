use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use vona_core::{
    AudioInputFrame, AudioOutputFrame, BackendCapabilities, BackendError, BackendStep,
    ExternalContextEvent, SessionConfig, SpeechToSpeechBackend,
};

use crate::onnx_runtime::SeamlessM4tOnnxRuntime;

const DEFAULT_MODEL_ID: &str = "facebook/hf-seamless-m4t-medium";
const DEFAULT_TARGET_LANG: &str = "eng";

#[derive(Debug)]
enum InboundControlKind {
    TranscriptOverride(String),
    PlanResult(String),
    PrecomputedReply(String),
    Unknown,
}

impl InboundControlKind {
    fn parse(event: &ExternalContextEvent) -> Self {
        match event.source.as_str() {
            "vona.transcript_override" => event
                .payload
                .as_str()
                .map(|s| InboundControlKind::TranscriptOverride(s.to_string()))
                .unwrap_or(InboundControlKind::Unknown),
            "vona.plan_result" | "vona.precomputed_reply" => event
                .spoken_summary
                .as_ref()
                .map(|s| {
                    if event.source == "vona.plan_result" {
                        InboundControlKind::PlanResult(s.trim().to_string())
                    } else {
                        InboundControlKind::PrecomputedReply(s.trim().to_string())
                    }
                })
                .unwrap_or(InboundControlKind::Unknown),
            _ => InboundControlKind::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeamlessM4tLocalConfig {
    pub model_id: String,
    pub target_language: String,
    pub source_language: Option<String>,
    pub speaker_id: u32,
    pub onnx_model_path: Option<String>,
    pub onnx_input_name: String,
    pub onnx_output_name: String,
    pub onnx_sample_rate_hz: u32,
}

impl Default for SeamlessM4tLocalConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            target_language: DEFAULT_TARGET_LANG.to_string(),
            source_language: Some(DEFAULT_TARGET_LANG.to_string()),
            speaker_id: 0,
            onnx_model_path: None,
            onnx_input_name: "audio".to_string(),
            onnx_output_name: "waveform".to_string(),
            onnx_sample_rate_hz: 16_000,
        }
    }
}

impl SeamlessM4tLocalConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(value) = std::env::var("VONA_STS_MODEL") {
            let value = value.trim();
            if !value.is_empty() {
                config.model_id = value.to_string();
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_TARGET_LANG") {
            let value = value.trim();
            if !value.is_empty() {
                config.target_language = value.to_string();
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_SOURCE_LANG") {
            let value = value.trim();
            if !value.is_empty() {
                config.source_language = Some(value.to_string());
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_SPEAKER_ID")
            && let Ok(parsed) = value.trim().parse()
        {
            config.speaker_id = parsed;
        }
        if let Ok(value) = std::env::var("VONA_STS_ONNX_MODEL_PATH") {
            let value = value.trim();
            if !value.is_empty() {
                config.onnx_model_path = Some(value.to_string());
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_ONNX_INPUT_NAME") {
            let value = value.trim();
            if !value.is_empty() {
                config.onnx_input_name = value.to_string();
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_ONNX_OUTPUT_NAME") {
            let value = value.trim();
            if !value.is_empty() {
                config.onnx_output_name = value.to_string();
            }
        }
        if let Ok(value) = std::env::var("VONA_STS_ONNX_SAMPLE_RATE")
            && let Ok(parsed) = value.trim().parse()
        {
            config.onnx_sample_rate_hz = parsed;
        }
        config
    }
}

#[derive(Debug, Clone)]
pub struct SeamlessM4tLocalSession {
    pub config: SessionConfig,
    pub pending_events: Vec<ExternalContextEvent>,
}

#[derive(Clone)]
pub struct SeamlessM4tLocalBackend {
    config: SeamlessM4tLocalConfig,
    runtime: Arc<SeamlessM4tOnnxRuntime>,
}

impl SeamlessM4tLocalBackend {
    pub fn new(config: SeamlessM4tLocalConfig) -> Result<Self, BackendError> {
        let runtime = SeamlessM4tOnnxRuntime::new(&config)?;
        Ok(Self {
            config,
            runtime: Arc::new(runtime),
        })
    }

    pub fn from_env() -> Result<Self, BackendError> {
        Self::new(SeamlessM4tLocalConfig::from_env())
    }

    fn extract_overrides(events: &[ExternalContextEvent]) -> (Option<String>, Option<String>) {
        let mut transcript_override = None;
        let mut reply_text = None;

        for event in events {
            match InboundControlKind::parse(event) {
                InboundControlKind::TranscriptOverride(text) => {
                    transcript_override = Some(text);
                }
                InboundControlKind::PlanResult(text)
                | InboundControlKind::PrecomputedReply(text) => {
                    reply_text = Some(text);
                }
                InboundControlKind::Unknown => {}
            }
        }

        (transcript_override, reply_text)
    }
}

#[async_trait]
impl SpeechToSpeechBackend for SeamlessM4tLocalBackend {
    type Session = SeamlessM4tLocalSession;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_context_injection: true,
            supports_style_conditioning: true,
            ..BackendCapabilities::default()
        }
    }

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, BackendError> {
        Ok(SeamlessM4tLocalSession {
            config,
            pending_events: Vec::new(),
        })
    }

    async fn step(
        &self,
        session: &mut Self::Session,
        input: AudioInputFrame,
    ) -> Result<BackendStep, BackendError> {
        let pending_events = std::mem::take(&mut session.pending_events);
        let (transcript_override, reply_text) = Self::extract_overrides(&pending_events);

        if reply_text.is_some() && input.samples.is_empty() {
            return Err(BackendError::Step(
                "ONNX local backend currently requires audio input for generation".to_string(),
            ));
        }

        let output_samples = self
            .runtime
            .run_audio_step(&input.samples, input.sample_rate_hz)
            .await?;

        Ok(BackendStep {
            output_audio: vec![AudioOutputFrame {
                sequence: input.sequence,
                sample_rate_hz: self.config.onnx_sample_rate_hz,
                channels: session.config.channels,
                samples: output_samples,
                is_filler: false,
            }],
            control_events: Vec::new(),
            transcript: transcript_override,
            finished: false,
            debug_payload: Some(json!({
                "backend_mode": "onnx",
                "reply_text": reply_text,
                "model_id": self.config.model_id,
                "onnx_model_path": self.config.onnx_model_path,
            })),
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
    use serde_json::json;
    use std::sync::Mutex;

    // Serialize all tests that mutate env vars to avoid race conditions.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── SeamlessM4tLocalConfig ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = SeamlessM4tLocalConfig::default();
        assert_eq!(cfg.model_id, "facebook/hf-seamless-m4t-medium");
        assert_eq!(cfg.target_language, "eng");
        assert_eq!(cfg.onnx_input_name, "audio");
        assert_eq!(cfg.onnx_output_name, "waveform");
        assert_eq!(cfg.onnx_sample_rate_hz, 16_000);
        assert_eq!(cfg.speaker_id, 0);
        assert!(cfg.onnx_model_path.is_none());
    }

    #[test]
    fn config_from_env_picks_up_model_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VONA_STS_ONNX_MODEL_PATH", "/tmp/test_model.onnx");
        }
        let cfg = SeamlessM4tLocalConfig::from_env();
        unsafe {
            std::env::remove_var("VONA_STS_ONNX_MODEL_PATH");
        }
        assert_eq!(cfg.onnx_model_path.as_deref(), Some("/tmp/test_model.onnx"));
    }

    #[test]
    fn config_from_env_ignores_blank_model_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VONA_STS_ONNX_MODEL_PATH", "   ");
        }
        let cfg = SeamlessM4tLocalConfig::from_env();
        unsafe {
            std::env::remove_var("VONA_STS_ONNX_MODEL_PATH");
        }
        assert!(cfg.onnx_model_path.is_none());
    }

    #[test]
    fn config_from_env_parses_sample_rate() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VONA_STS_ONNX_SAMPLE_RATE", "48000");
        }
        let cfg = SeamlessM4tLocalConfig::from_env();
        unsafe {
            std::env::remove_var("VONA_STS_ONNX_SAMPLE_RATE");
        }
        assert_eq!(cfg.onnx_sample_rate_hz, 48_000);
    }

    // ── SeamlessM4tLocalBackend::new() ─────────────────────────────────────

    #[test]
    fn backend_new_fails_without_model_path() {
        let cfg = SeamlessM4tLocalConfig::default(); // onnx_model_path = None
        let result = SeamlessM4tLocalBackend::new(cfg);
        assert!(result.is_err(), "expected Err when onnx_model_path is None");
        if let Err(BackendError::Start(msg)) = result {
            assert!(
                msg.contains("VONA_STS_ONNX_MODEL_PATH"),
                "error message should mention the env var, got: {msg}"
            );
        }
    }

    #[test]
    fn backend_new_fails_with_nonexistent_model_file() {
        let cfg = SeamlessM4tLocalConfig {
            onnx_model_path: Some("/nonexistent/path/model.onnx".into()),
            ..Default::default()
        };
        let result = SeamlessM4tLocalBackend::new(cfg);
        assert!(result.is_err(), "expected Err for missing ONNX file");
    }

    // ── extract_overrides ──────────────────────────────────────────────────

    #[test]
    fn extract_overrides_returns_none_for_empty_events() {
        let (transcript, reply) = SeamlessM4tLocalBackend::extract_overrides(&[]);
        assert!(transcript.is_none());
        assert!(reply.is_none());
    }

    #[test]
    fn extract_overrides_parses_transcript_override() {
        let events = vec![ExternalContextEvent {
            source: "vona.transcript_override".into(),
            spoken_summary: None,
            payload: json!("hello world"),
        }];
        let (transcript, reply) = SeamlessM4tLocalBackend::extract_overrides(&events);
        assert_eq!(transcript.as_deref(), Some("hello world"));
        assert!(reply.is_none());
    }

    #[test]
    fn extract_overrides_parses_plan_result() {
        let events = vec![ExternalContextEvent {
            source: "vona.plan_result".into(),
            spoken_summary: Some("Here is the result.".into()),
            payload: json!(null),
        }];
        let (transcript, reply) = SeamlessM4tLocalBackend::extract_overrides(&events);
        assert!(transcript.is_none());
        assert_eq!(reply.as_deref(), Some("Here is the result."));
    }

    #[test]
    fn extract_overrides_parses_precomputed_reply() {
        let events = vec![ExternalContextEvent {
            source: "vona.precomputed_reply".into(),
            spoken_summary: Some("Precomputed text.".into()),
            payload: json!(null),
        }];
        let (transcript, reply) = SeamlessM4tLocalBackend::extract_overrides(&events);
        assert!(transcript.is_none());
        assert_eq!(reply.as_deref(), Some("Precomputed text."));
    }

    #[test]
    fn extract_overrides_unknown_source_ignored() {
        let events = vec![ExternalContextEvent {
            source: "unknown.source".into(),
            spoken_summary: Some("ignored".into()),
            payload: json!(null),
        }];
        let (transcript, reply) = SeamlessM4tLocalBackend::extract_overrides(&events);
        assert!(transcript.is_none());
        assert!(reply.is_none());
    }
}
