use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub const DEFAULT_API_BASE: &str = "https://api.deepgram.com";
pub const DEFAULT_STT_MODEL: &str = "flux-general-en";
pub const DEFAULT_TTS_MODEL: &str = "aura-2-thalia-en";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepgramConfig {
    pub api_base: String,
    pub api_key: Option<String>,
}

impl Default for DeepgramConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
        }
    }
}

impl DeepgramConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("DEEPGRAM_API_BASE")
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("DEEPGRAM_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepgramSttConfig {
    pub base: DeepgramConfig,
    pub model: String,
    pub endpoint_version: String,
    pub encoding: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub interim_results: bool,
}

impl Default for DeepgramSttConfig {
    fn default() -> Self {
        Self {
            base: DeepgramConfig::default(),
            model: DEFAULT_STT_MODEL.to_string(),
            endpoint_version: "v2".to_string(),
            encoding: "linear16".to_string(),
            sample_rate_hz: 16_000,
            channels: 1,
            interim_results: true,
        }
    }
}

impl DeepgramSttConfig {
    pub fn websocket_url(&self) -> String {
        let base = self
            .base
            .api_base
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!(
            "{base}/{}/listen?model={}&encoding={}&sample_rate={}&channels={}&interim_results={}",
            self.endpoint_version,
            self.model,
            self.encoding,
            self.sample_rate_hz,
            self.channels,
            self.interim_results
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepgramTtsConfig {
    pub base: DeepgramConfig,
    pub model: String,
    pub encoding: String,
    pub sample_rate_hz: u32,
}

impl Default for DeepgramTtsConfig {
    fn default() -> Self {
        Self {
            base: DeepgramConfig::default(),
            model: DEFAULT_TTS_MODEL.to_string(),
            encoding: "linear16".to_string(),
            sample_rate_hz: 24_000,
        }
    }
}

impl DeepgramTtsConfig {
    pub fn websocket_url(&self) -> String {
        let base = self
            .base
            .api_base
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!(
            "{base}/v1/speak?model={}&encoding={}&sample_rate={}",
            self.model, self.encoding, self.sample_rate_hz
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepgramTtsMessage {
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum DeepgramMappingError {
    #[error("text cannot be empty")]
    EmptyText,
}

pub fn tts_text_message(
    text: impl Into<String>,
) -> Result<DeepgramTtsMessage, DeepgramMappingError> {
    let text = text.into();
    if text.is_empty() {
        return Err(DeepgramMappingError::EmptyText);
    }
    Ok(DeepgramTtsMessage {
        payload: json!({ "type": "Speak", "text": text }),
    })
}

pub fn tts_flush_message() -> DeepgramTtsMessage {
    DeepgramTtsMessage {
        payload: json!({ "type": "Flush" }),
    }
}

pub fn tts_close_message() -> DeepgramTtsMessage {
    DeepgramTtsMessage {
        payload: json!({ "type": "Close" }),
    }
}

pub fn transcript_from_listen_message(message: &Value) -> Option<String> {
    message
        .pointer("/channel/alternatives/0/transcript")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_url_targets_listen_websocket() {
        let cfg = DeepgramSttConfig {
            base: DeepgramConfig {
                api_base: "https://example.test".to_string(),
                api_key: None,
            },
            ..DeepgramSttConfig::default()
        };
        assert_eq!(
            cfg.websocket_url(),
            "wss://example.test/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000&channels=1&interim_results=true"
        );
    }

    #[test]
    fn tts_url_targets_speak_websocket() {
        let cfg = DeepgramTtsConfig {
            base: DeepgramConfig {
                api_base: "https://example.test".to_string(),
                api_key: None,
            },
            ..DeepgramTtsConfig::default()
        };
        assert_eq!(
            cfg.websocket_url(),
            "wss://example.test/v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000"
        );
    }

    #[test]
    fn transcript_parser_ignores_empty_transcripts() {
        let message = json!({ "channel": { "alternatives": [{ "transcript": "" }] } });
        assert_eq!(transcript_from_listen_message(&message), None);
    }

    #[test]
    fn transcript_parser_reads_first_alternative() {
        let message = json!({ "channel": { "alternatives": [{ "transcript": "hello" }] } });
        assert_eq!(
            transcript_from_listen_message(&message),
            Some("hello".to_string())
        );
    }
}
