use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use vona_core::{
    AudioInputFrame, RealtimeVoiceCapabilities, RealtimeVoiceInput, RealtimeVoiceModelFamily,
    RealtimeVoiceOutput, RealtimeVoiceSessionConfig,
};

pub const DEFAULT_API_BASE: &str = "https://generativelanguage.googleapis.com";
pub const DEFAULT_API_VERSION: &str = "v1alpha";
pub const DEFAULT_MODEL: &str = "gemini-2.5-flash-native-audio-preview-12-2025";
pub const DEFAULT_VOICE: &str = "Kore";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiLiveConfig {
    pub api_base: String,
    pub api_key: Option<String>,
    pub api_version: String,
    pub model: String,
    pub voice: String,
    pub input_sample_rate_hz: u32,
}

impl Default for GeminiLiveConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
            api_version: DEFAULT_API_VERSION.to_string(),
            model: DEFAULT_MODEL.to_string(),
            voice: DEFAULT_VOICE.to_string(),
            input_sample_rate_hz: 16_000,
        }
    }
}

impl GeminiLiveConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("GEMINI_API_BASE")
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("GEMINI_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            api_version: std::env::var("GEMINI_LIVE_API_VERSION")
                .unwrap_or_else(|_| DEFAULT_API_VERSION.to_string()),
            model: std::env::var("GEMINI_LIVE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            voice: std::env::var("GEMINI_LIVE_VOICE").unwrap_or_else(|_| DEFAULT_VOICE.to_string()),
            input_sample_rate_hz: std::env::var("GEMINI_LIVE_INPUT_SAMPLE_RATE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16_000),
        }
    }

    pub fn websocket_url(&self) -> String {
        let base = self
            .api_base
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!(
            "{base}/ws/google.ai.generativelanguage.{}/GenerativeService.BidiGenerateContent",
            self.api_version
        )
    }

    pub fn session_config(&self, session_id: impl Into<String>) -> RealtimeVoiceSessionConfig {
        RealtimeVoiceSessionConfig {
            session_id: session_id.into(),
            input_sample_rate_hz: self.input_sample_rate_hz,
            output_sample_rate_hz: 24_000,
            channels: 1,
            model_family: RealtimeVoiceModelFamily::HostedRealtimeApi {
                provider: "gemini".to_string(),
                model: self.model.clone(),
            },
            metadata: json!({
                "voice": self.voice,
                "api_version": self.api_version,
            }),
        }
    }

    pub fn capabilities(&self) -> RealtimeVoiceCapabilities {
        RealtimeVoiceCapabilities {
            supports_full_duplex: true,
            supports_streaming_audio_input: true,
            supports_streaming_audio_output: true,
            supports_tool_calls: true,
            supports_interruption: true,
            supports_context_injection: true,
            is_hosted_service: true,
            max_input_chunk_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiLiveClientMessage {
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum GeminiLiveMappingError {
    #[error("Gemini Live does not accept Vona event: {0}")]
    UnsupportedInput(String),
    #[error("Gemini Live server message is missing required field: {0}")]
    MissingField(&'static str),
}

pub fn setup_message(config: &GeminiLiveConfig) -> GeminiLiveClientMessage {
    GeminiLiveClientMessage {
        payload: json!({
            "setup": {
                "model": format!("models/{}", config.model),
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": { "voiceName": config.voice }
                        }
                    }
                }
            }
        }),
    }
}

pub fn input_to_client_message(
    input: RealtimeVoiceInput,
    sample_rate_hz: u32,
) -> Result<GeminiLiveClientMessage, GeminiLiveMappingError> {
    match input {
        RealtimeVoiceInput::Audio(frame) => Ok(audio_message(&frame, sample_rate_hz)),
        RealtimeVoiceInput::Text { text } => Ok(GeminiLiveClientMessage {
            payload: json!({
                "clientContent": {
                    "turns": [{ "role": "user", "parts": [{ "text": text }] }],
                    "turnComplete": true
                }
            }),
        }),
        RealtimeVoiceInput::Control(vona_core::RealtimeVoiceControl::CommitInput) => {
            Ok(GeminiLiveClientMessage {
                payload: json!({ "clientContent": { "turnComplete": true } }),
            })
        }
        RealtimeVoiceInput::Control(control) => Err(GeminiLiveMappingError::UnsupportedInput(
            format!("{control:?}"),
        )),
        RealtimeVoiceInput::ToolResult(event) => Ok(GeminiLiveClientMessage {
            payload: json!({
                "toolResponse": {
                    "functionResponses": [{
                        "name": event.source,
                        "response": event.payload,
                    }]
                }
            }),
        }),
    }
}

pub fn audio_message(frame: &AudioInputFrame, sample_rate_hz: u32) -> GeminiLiveClientMessage {
    GeminiLiveClientMessage {
        payload: json!({
            "realtimeInput": {
                "mediaChunks": [{
                    "mimeType": format!("audio/pcm;rate={sample_rate_hz}"),
                    "data": base64::engine::general_purpose::STANDARD.encode(samples_to_pcm16_le(&frame.samples)),
                }]
            }
        }),
    }
}

pub fn server_message_to_output(
    message: &Value,
) -> Result<Option<RealtimeVoiceOutput>, GeminiLiveMappingError> {
    let Some(parts) = message
        .pointer("/serverContent/modelTurn/parts")
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };

    for part in parts {
        if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) {
            let pcm = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|_| GeminiLiveMappingError::MissingField("inlineData.data"))?;
            return Ok(Some(RealtimeVoiceOutput::Audio(
                vona_core::AudioOutputFrame {
                    sequence: 0,
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: pcm16_le_to_samples(&pcm),
                    is_filler: false,
                },
            )));
        }
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            return Ok(Some(RealtimeVoiceOutput::TranscriptFragment {
                text: text.to_string(),
                final_fragment: false,
            }));
        }
    }
    Ok(None)
}

fn samples_to_pcm16_le(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| {
            let sample = sample.clamp(-1.0, 1.0);
            let pcm = if sample < 0.0 {
                (sample * 32768.0).round() as i16
            } else {
                (sample * 32767.0).round() as i16
            };
            pcm.to_le_bytes()
        })
        .collect()
}

fn pcm16_le_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_message_uses_native_audio_model_and_voice() {
        let message = setup_message(&GeminiLiveConfig::default());
        assert_eq!(
            message.payload["setup"]["model"],
            "models/gemini-2.5-flash-native-audio-preview-12-2025"
        );
        assert_eq!(
            message.payload["setup"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
                ["voiceName"],
            "Kore"
        );
    }

    #[test]
    fn audio_message_includes_pcm_rate_mime_type() {
        let message = audio_message(
            &AudioInputFrame {
                sequence: 1,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0.0, 1.0, -1.0],
            },
            16_000,
        );
        assert_eq!(
            message.payload["realtimeInput"]["mediaChunks"][0]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert_eq!(
            message.payload["realtimeInput"]["mediaChunks"][0]["data"],
            "AAD/fwCA"
        );
    }

    #[test]
    fn server_inline_audio_decodes_to_vona_output() {
        let message = json!({
            "serverContent": {
                "modelTurn": {
                    "parts": [{ "inlineData": { "mimeType": "audio/pcm;rate=24000", "data": "AAD/fwCA" } }]
                }
            }
        });
        let output = server_message_to_output(&message).unwrap().unwrap();
        assert_eq!(
            output,
            RealtimeVoiceOutput::Audio(vona_core::AudioOutputFrame {
                sequence: 0,
                sample_rate_hz: 24_000,
                channels: 1,
                samples: vec![0.0, 32767.0 / 32768.0, -1.0],
                is_filler: false,
            })
        );
    }
}
