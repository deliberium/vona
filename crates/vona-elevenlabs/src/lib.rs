use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub const DEFAULT_API_BASE: &str = "https://api.elevenlabs.io";
pub const DEFAULT_MODEL: &str = "eleven_flash_v2_5";
pub const DEFAULT_OUTPUT_FORMAT: &str = "pcm_24000";

#[derive(Debug, Clone, PartialEq)]
pub struct ElevenLabsTtsConfig {
    pub api_base: String,
    pub api_key: Option<String>,
    pub voice_id: String,
    pub model_id: String,
    pub output_format: String,
    pub stability: Option<f32>,
    pub similarity_boost: Option<f32>,
    pub speed: Option<f32>,
}

impl Default for ElevenLabsTtsConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
            voice_id: "JBFqnCBsd6RMkjVDRZzb".to_string(),
            model_id: DEFAULT_MODEL.to_string(),
            output_format: DEFAULT_OUTPUT_FORMAT.to_string(),
            stability: Some(0.5),
            similarity_boost: Some(0.8),
            speed: Some(1.0),
        }
    }
}

impl ElevenLabsTtsConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("ELEVENLABS_API_BASE")
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("ELEVENLABS_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            voice_id: std::env::var("ELEVENLABS_VOICE_ID")
                .unwrap_or_else(|_| "JBFqnCBsd6RMkjVDRZzb".to_string()),
            model_id: std::env::var("ELEVENLABS_MODEL_ID")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            output_format: std::env::var("ELEVENLABS_OUTPUT_FORMAT")
                .unwrap_or_else(|_| DEFAULT_OUTPUT_FORMAT.to_string()),
            stability: parse_optional_f32("ELEVENLABS_STABILITY").or(Some(0.5)),
            similarity_boost: parse_optional_f32("ELEVENLABS_SIMILARITY_BOOST").or(Some(0.8)),
            speed: parse_optional_f32("ELEVENLABS_SPEED").or(Some(1.0)),
        }
    }

    pub fn websocket_url(&self) -> String {
        let base = self
            .api_base
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!(
            "{base}/v1/text-to-speech/{}/stream-input?model_id={}&output_format={}",
            self.voice_id, self.model_id, self.output_format
        )
    }

    pub fn streaming_http_url(&self) -> String {
        format!(
            "{}/v1/text-to-speech/{}/stream?output_format={}",
            self.api_base.trim_end_matches('/'),
            self.voice_id,
            self.output_format
        )
    }

    pub fn voice_settings(&self) -> Value {
        let mut settings = serde_json::Map::new();
        if let Some(value) = self.stability {
            settings.insert("stability".to_string(), json!(value));
        }
        if let Some(value) = self.similarity_boost {
            settings.insert("similarity_boost".to_string(), json!(value));
        }
        if let Some(value) = self.speed {
            settings.insert("speed".to_string(), json!(value));
        }
        Value::Object(settings)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElevenLabsWebSocketMessage {
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ElevenLabsMappingError {
    #[error("text cannot be empty unless it is the end-of-stream marker")]
    EmptyText,
}

pub fn start_message(config: &ElevenLabsTtsConfig) -> ElevenLabsWebSocketMessage {
    let mut payload = serde_json::Map::new();
    payload.insert("text".to_string(), json!(" "));
    payload.insert("voice_settings".to_string(), config.voice_settings());
    if let Some(api_key) = &config.api_key {
        payload.insert("xi_api_key".to_string(), json!(api_key));
    }
    ElevenLabsWebSocketMessage {
        payload: Value::Object(payload),
    }
}

pub fn text_message(
    text: impl Into<String>,
) -> Result<ElevenLabsWebSocketMessage, ElevenLabsMappingError> {
    let text = text.into();
    if text.is_empty() {
        return Err(ElevenLabsMappingError::EmptyText);
    }
    Ok(ElevenLabsWebSocketMessage {
        payload: json!({
            "text": text,
            "try_trigger_generation": true,
        }),
    })
}

pub fn end_message() -> ElevenLabsWebSocketMessage {
    ElevenLabsWebSocketMessage {
        payload: json!({ "text": "" }),
    }
}

fn parse_optional_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn websocket_url_uses_stream_input_endpoint() {
        let cfg = ElevenLabsTtsConfig {
            api_base: "https://example.test".to_string(),
            voice_id: "voice".to_string(),
            model_id: "eleven_flash_v2_5".to_string(),
            output_format: "pcm_24000".to_string(),
            ..ElevenLabsTtsConfig::default()
        };
        assert_eq!(
            cfg.websocket_url(),
            "wss://example.test/v1/text-to-speech/voice/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
        );
    }

    #[test]
    fn text_message_sets_generation_trigger() {
        let message = text_message("hello").unwrap();
        assert_eq!(message.payload["text"], "hello");
        assert_eq!(message.payload["try_trigger_generation"], true);
    }

    #[test]
    fn end_message_is_empty_text_marker() {
        assert_eq!(end_message().payload["text"], "");
    }
}
