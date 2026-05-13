use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use vona_core::{
    AudioInputFrame, RealtimeVoiceCapabilities, RealtimeVoiceInput, RealtimeVoiceModelFamily,
    RealtimeVoiceOutput, RealtimeVoiceSessionConfig,
};

pub const DEFAULT_API_BASE: &str = "https://api.openai.com";
pub const DEFAULT_MODEL: &str = "gpt-realtime";
pub const DEFAULT_VOICE: &str = "marin";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiRealtimeConfig {
    pub api_base: String,
    pub api_key: Option<String>,
    pub model: String,
    pub voice: String,
    pub input_audio_format: String,
    pub output_audio_format: String,
}

impl Default for OpenAiRealtimeConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
            model: DEFAULT_MODEL.to_string(),
            voice: DEFAULT_VOICE.to_string(),
            input_audio_format: "pcm16".to_string(),
            output_audio_format: "pcm16".to_string(),
        }
    }
}

impl OpenAiRealtimeConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|value| !value.is_empty()),
            model: std::env::var("OPENAI_REALTIME_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            voice: std::env::var("OPENAI_REALTIME_VOICE")
                .unwrap_or_else(|_| DEFAULT_VOICE.to_string()),
            input_audio_format: std::env::var("OPENAI_REALTIME_INPUT_AUDIO_FORMAT")
                .unwrap_or_else(|_| "pcm16".to_string()),
            output_audio_format: std::env::var("OPENAI_REALTIME_OUTPUT_AUDIO_FORMAT")
                .unwrap_or_else(|_| "pcm16".to_string()),
        }
    }

    pub fn websocket_url(&self) -> String {
        let base = self
            .api_base
            .trim_end_matches('/')
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        format!("{base}/v1/realtime?model={}", self.model)
    }

    pub fn session_config(&self, session_id: impl Into<String>) -> RealtimeVoiceSessionConfig {
        RealtimeVoiceSessionConfig {
            session_id: session_id.into(),
            input_sample_rate_hz: 24_000,
            output_sample_rate_hz: 24_000,
            channels: 1,
            model_family: RealtimeVoiceModelFamily::HostedRealtimeApi {
                provider: "openai".to_string(),
                model: self.model.clone(),
            },
            metadata: json!({
                "voice": self.voice,
                "input_audio_format": self.input_audio_format,
                "output_audio_format": self.output_audio_format,
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
pub struct OpenAiClientEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum OpenAiRealtimeMappingError {
    #[error("OpenAI Realtime does not accept Vona event: {0}")]
    UnsupportedInput(String),
    #[error("OpenAI Realtime server event is missing required field: {0}")]
    MissingField(&'static str),
}

pub fn session_update_event(config: &OpenAiRealtimeConfig) -> OpenAiClientEvent {
    OpenAiClientEvent {
        event_type: "session.update".to_string(),
        payload: json!({
            "session": {
                "type": "realtime",
                "model": config.model,
                "audio": {
                    "input": { "format": config.input_audio_format },
                    "output": {
                        "format": config.output_audio_format,
                        "voice": config.voice,
                    }
                }
            }
        }),
    }
}

pub fn input_to_client_event(
    input: RealtimeVoiceInput,
) -> Result<OpenAiClientEvent, OpenAiRealtimeMappingError> {
    match input {
        RealtimeVoiceInput::Audio(frame) => Ok(audio_append_event(&frame)),
        RealtimeVoiceInput::Text { text } => Ok(OpenAiClientEvent {
            event_type: "conversation.item.create".to_string(),
            payload: json!({
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }]
                }
            }),
        }),
        RealtimeVoiceInput::Control(control) => Ok(OpenAiClientEvent {
            event_type: match control {
                vona_core::RealtimeVoiceControl::StartResponse => "response.create",
                vona_core::RealtimeVoiceControl::CommitInput => "input_audio_buffer.commit",
                vona_core::RealtimeVoiceControl::ClearOutput => "input_audio_buffer.clear",
                vona_core::RealtimeVoiceControl::Interrupt { .. } => "response.cancel",
                vona_core::RealtimeVoiceControl::Close => "session.close",
            }
            .to_string(),
            payload: json!({}),
        }),
        RealtimeVoiceInput::ToolResult(event) => Ok(OpenAiClientEvent {
            event_type: "conversation.item.create".to_string(),
            payload: json!({
                "item": {
                    "type": "function_call_output",
                    "call_id": event.source,
                    "output": event.payload.to_string()
                }
            }),
        }),
    }
}

pub fn audio_append_event(frame: &AudioInputFrame) -> OpenAiClientEvent {
    OpenAiClientEvent {
        event_type: "input_audio_buffer.append".to_string(),
        payload: json!({
            "audio": base64::engine::general_purpose::STANDARD.encode(samples_to_pcm16_le(&frame.samples)),
        }),
    }
}

pub fn server_event_to_output(
    event: &Value,
) -> Result<Option<RealtimeVoiceOutput>, OpenAiRealtimeMappingError> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .ok_or(OpenAiRealtimeMappingError::MissingField("type"))?;

    match event_type {
        "response.output_audio.delta" => {
            let audio = event
                .get("delta")
                .and_then(Value::as_str)
                .ok_or(OpenAiRealtimeMappingError::MissingField("delta"))?;
            let pcm = base64::engine::general_purpose::STANDARD
                .decode(audio)
                .map_err(|_| OpenAiRealtimeMappingError::MissingField("delta"))?;
            Ok(Some(RealtimeVoiceOutput::Audio(
                vona_core::AudioOutputFrame {
                    sequence: 0,
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: pcm16_le_to_samples(&pcm),
                    is_filler: false,
                },
            )))
        }
        "response.output_audio_transcript.delta" | "response.output_text.delta" => {
            let text = event
                .get("delta")
                .and_then(Value::as_str)
                .ok_or(OpenAiRealtimeMappingError::MissingField("delta"))?;
            Ok(Some(RealtimeVoiceOutput::TranscriptFragment {
                text: text.to_string(),
                final_fragment: false,
            }))
        }
        "response.done" => Ok(Some(RealtimeVoiceOutput::Closed {
            reason: Some("response.done".to_string()),
        })),
        _ => Ok(None),
    }
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
    use vona_core::RealtimeVoiceControl;

    #[test]
    fn websocket_url_uses_realtime_model_query() {
        let cfg = OpenAiRealtimeConfig {
            api_base: "https://example.test/".to_string(),
            model: "gpt-realtime".to_string(),
            ..OpenAiRealtimeConfig::default()
        };
        assert_eq!(
            cfg.websocket_url(),
            "wss://example.test/v1/realtime?model=gpt-realtime"
        );
    }

    #[test]
    fn audio_input_maps_to_base64_append_event() {
        let event = input_to_client_event(RealtimeVoiceInput::Audio(AudioInputFrame {
            sequence: 7,
            sample_rate_hz: 24_000,
            channels: 1,
            samples: vec![0.0, 1.0, -1.0],
        }))
        .unwrap();
        assert_eq!(event.event_type, "input_audio_buffer.append");
        assert_eq!(event.payload["audio"], "AAD/fwCA");
    }

    #[test]
    fn interrupt_maps_to_response_cancel() {
        let event = input_to_client_event(RealtimeVoiceInput::Control(
            RealtimeVoiceControl::Interrupt {
                reason: Some("barge-in".to_string()),
            },
        ))
        .unwrap();
        assert_eq!(event.event_type, "response.cancel");
    }

    #[test]
    fn output_audio_delta_decodes_to_vona_audio() {
        let event = json!({ "type": "response.output_audio.delta", "delta": "AAD/fwCA" });
        let output = server_event_to_output(&event).unwrap().unwrap();
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
