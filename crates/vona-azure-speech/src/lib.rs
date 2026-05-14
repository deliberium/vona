use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub const DEFAULT_REGION: &str = "eastus";
pub const DEFAULT_LANGUAGE: &str = "en-US";
pub const DEFAULT_VOICE: &str = "en-US-AvaMultilingualNeural";
pub const DEFAULT_VOICE_LIVE_API_VERSION: &str = "2025-10-01";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureSpeechConfig {
    pub region: String,
    pub subscription_key: Option<String>,
    pub language: String,
    pub voice: String,
    pub output_format: String,
}

impl Default for AzureSpeechConfig {
    fn default() -> Self {
        Self {
            region: DEFAULT_REGION.to_string(),
            subscription_key: None,
            language: DEFAULT_LANGUAGE.to_string(),
            voice: DEFAULT_VOICE.to_string(),
            output_format: "raw-24khz-16bit-mono-pcm".to_string(),
        }
    }
}

impl AzureSpeechConfig {
    pub fn from_env() -> Self {
        Self {
            region: std::env::var("AZURE_SPEECH_REGION")
                .unwrap_or_else(|_| DEFAULT_REGION.to_string()),
            subscription_key: std::env::var("AZURE_SPEECH_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            language: std::env::var("AZURE_SPEECH_LANGUAGE")
                .unwrap_or_else(|_| DEFAULT_LANGUAGE.to_string()),
            voice: std::env::var("AZURE_SPEECH_VOICE")
                .unwrap_or_else(|_| DEFAULT_VOICE.to_string()),
            output_format: std::env::var("AZURE_SPEECH_OUTPUT_FORMAT")
                .unwrap_or_else(|_| "raw-24khz-16bit-mono-pcm".to_string()),
        }
    }

    pub fn speech_to_text_websocket_url(&self) -> String {
        format!(
            "wss://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={}",
            self.region, self.language
        )
    }

    pub fn text_to_speech_websocket_url(&self) -> String {
        format!(
            "wss://{}.tts.speech.microsoft.com/cognitiveservices/websocket/v1",
            self.region
        )
    }

    pub fn ssml(&self, text: impl AsRef<str>) -> Result<String, AzureSpeechMappingError> {
        let text = text.as_ref();
        if text.is_empty() {
            return Err(AzureSpeechMappingError::EmptyText);
        }
        Ok(format!(
            "<speak version=\"1.0\" xml:lang=\"{}\"><voice name=\"{}\">{}</voice></speak>",
            self.language,
            self.voice,
            escape_xml(text)
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureVoiceLiveConfig {
    pub resource_name: String,
    pub api_key: Option<String>,
    pub api_version: String,
    pub use_foundry_services_host: bool,
}

impl AzureVoiceLiveConfig {
    pub fn from_env() -> Option<Self> {
        Some(Self {
            resource_name: std::env::var("AZURE_VOICE_LIVE_RESOURCE").ok()?,
            api_key: std::env::var("AZURE_VOICE_LIVE_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            api_version: std::env::var("AZURE_VOICE_LIVE_API_VERSION")
                .unwrap_or_else(|_| DEFAULT_VOICE_LIVE_API_VERSION.to_string()),
            use_foundry_services_host: std::env::var("AZURE_VOICE_LIVE_FOUNDRY_HOST")
                .ok()
                .as_deref()
                == Some("1"),
        })
    }

    pub fn websocket_url(&self) -> String {
        let host = if self.use_foundry_services_host {
            "services.ai.azure.com"
        } else {
            "cognitiveservices.azure.com"
        };
        format!(
            "wss://{}.{}/voice-live/realtime?api-version={}",
            self.resource_name, host, self.api_version
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AzureSpeechMessage {
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AzureSpeechMappingError {
    #[error("text cannot be empty")]
    EmptyText,
}

pub fn speech_config_message(config: &AzureSpeechConfig) -> AzureSpeechMessage {
    AzureSpeechMessage {
        payload: json!({
            "context": {
                "synthesis": {
                    "audio": {
                        "metadataoptions": {
                            "sentenceBoundaryEnabled": false,
                            "wordBoundaryEnabled": false,
                        },
                        "outputFormat": config.output_format,
                    }
                }
            }
        }),
    }
}

pub fn transcript_from_recognition_message(message: &Value) -> Option<String> {
    message
        .get("DisplayText")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_url_uses_region_and_language() {
        let cfg = AzureSpeechConfig {
            region: "uksouth".to_string(),
            language: "en-GB".to_string(),
            ..AzureSpeechConfig::default()
        };
        assert_eq!(
            cfg.speech_to_text_websocket_url(),
            "wss://uksouth.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-GB"
        );
    }

    #[test]
    fn tts_url_uses_region() {
        let cfg = AzureSpeechConfig {
            region: "uksouth".to_string(),
            ..AzureSpeechConfig::default()
        };
        assert_eq!(
            cfg.text_to_speech_websocket_url(),
            "wss://uksouth.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
        );
    }

    #[test]
    fn ssml_escapes_text() {
        let cfg = AzureSpeechConfig::default();
        let ssml = cfg.ssml("hello <world> & \"you\"").unwrap();
        assert!(ssml.contains("hello &lt;world&gt; &amp; &quot;you&quot;"));
    }

    #[test]
    fn transcript_parser_reads_display_text() {
        let message = json!({ "DisplayText": "hello" });
        assert_eq!(
            transcript_from_recognition_message(&message),
            Some("hello".to_string())
        );
    }

    #[test]
    fn voice_live_url_supports_foundry_host() {
        let cfg = AzureVoiceLiveConfig {
            resource_name: "my-foundry-resource".to_string(),
            api_key: None,
            api_version: "2025-10-01".to_string(),
            use_foundry_services_host: true,
        };
        assert_eq!(
            cfg.websocket_url(),
            "wss://my-foundry-resource.services.ai.azure.com/voice-live/realtime?api-version=2025-10-01"
        );
    }
}
