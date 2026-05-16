use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use vona_core::{AudioInputFrame, AudioOutputFrame};

pub const DEFAULT_API_BASE: &str = "https://dashscope.aliyuncs.com";
pub const DEFAULT_ASR_MODEL: &str = "qwen3-asr-flash-realtime";
pub const DEFAULT_TTS_MODEL: &str = "qwen3-tts-flash-realtime";
pub const DEFAULT_VOICE: &str = "Cherry";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAsrRealtimeConfig {
    pub api_base: String,
    pub api_key: Option<String>,
    pub model: String,
    pub input_audio_format: String,
    pub sample_rate_hz: u32,
    pub language: Option<String>,
}

impl Default for QwenAsrRealtimeConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
            model: DEFAULT_ASR_MODEL.to_string(),
            input_audio_format: "pcm16".to_string(),
            sample_rate_hz: 16_000,
            language: None,
        }
    }
}

impl QwenAsrRealtimeConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("QWEN_API_BASE")
                .or_else(|_| std::env::var("DASHSCOPE_API_BASE"))
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("QWEN_API_KEY")
                .or_else(|_| std::env::var("DASHSCOPE_API_KEY"))
                .ok()
                .filter(|value| !value.is_empty()),
            model: std::env::var("QWEN_ASR_MODEL")
                .unwrap_or_else(|_| DEFAULT_ASR_MODEL.to_string()),
            input_audio_format: std::env::var("QWEN_ASR_INPUT_AUDIO_FORMAT")
                .unwrap_or_else(|_| "pcm16".to_string()),
            sample_rate_hz: std::env::var("QWEN_ASR_SAMPLE_RATE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16_000),
            language: std::env::var("QWEN_ASR_LANGUAGE")
                .ok()
                .filter(|value| !value.is_empty()),
        }
    }

    pub fn websocket_url(&self) -> String {
        realtime_websocket_url(&self.api_base, &self.model)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenTtsRealtimeConfig {
    pub api_base: String,
    pub api_key: Option<String>,
    pub model: String,
    pub voice: String,
    pub input_text_format: String,
    pub output_audio_format: String,
    pub sample_rate_hz: u32,
}

impl Default for QwenTtsRealtimeConfig {
    fn default() -> Self {
        Self {
            api_base: DEFAULT_API_BASE.to_string(),
            api_key: None,
            model: DEFAULT_TTS_MODEL.to_string(),
            voice: DEFAULT_VOICE.to_string(),
            input_text_format: "text".to_string(),
            output_audio_format: "pcm".to_string(),
            sample_rate_hz: 24_000,
        }
    }
}

impl QwenTtsRealtimeConfig {
    pub fn from_env() -> Self {
        Self {
            api_base: std::env::var("QWEN_API_BASE")
                .or_else(|_| std::env::var("DASHSCOPE_API_BASE"))
                .unwrap_or_else(|_| DEFAULT_API_BASE.to_string()),
            api_key: std::env::var("QWEN_API_KEY")
                .or_else(|_| std::env::var("DASHSCOPE_API_KEY"))
                .ok()
                .filter(|value| !value.is_empty()),
            model: std::env::var("QWEN_TTS_MODEL")
                .unwrap_or_else(|_| DEFAULT_TTS_MODEL.to_string()),
            voice: std::env::var("QWEN_TTS_VOICE").unwrap_or_else(|_| DEFAULT_VOICE.to_string()),
            input_text_format: std::env::var("QWEN_TTS_INPUT_TEXT_FORMAT")
                .unwrap_or_else(|_| "text".to_string()),
            output_audio_format: std::env::var("QWEN_TTS_OUTPUT_AUDIO_FORMAT")
                .unwrap_or_else(|_| "pcm".to_string()),
            sample_rate_hz: std::env::var("QWEN_TTS_SAMPLE_RATE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(24_000),
        }
    }

    pub fn websocket_url(&self) -> String {
        realtime_websocket_url(&self.api_base, &self.model)
    }
}

fn realtime_websocket_url(api_base: &str, model: &str) -> String {
    let base = api_base
        .trim_end_matches('/')
        .replace("https://", "wss://")
        .replace("http://", "ws://");
    format!("{base}/api-ws/v1/realtime?model={model}")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenClientEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QwenRealtimeServerOutput {
    TranscriptFragment {
        text: String,
        final_fragment: bool,
    },
    Audio(AudioOutputFrame),
    Completed {
        reason: Option<String>,
    },
    Error {
        code: Option<String>,
        message: String,
    },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum QwenRealtimeMappingError {
    #[error("Qwen realtime event is missing required field: {0}")]
    MissingField(&'static str),
    #[error("Qwen realtime event contains invalid audio payload")]
    InvalidAudio,
}

pub fn qwen_session_update_event_for_asr(config: &QwenAsrRealtimeConfig) -> QwenClientEvent {
    let mut transcription = json!({});
    if let Some(language) = &config.language {
        transcription["language"] = json!(language);
    }
    qwen_session_update_event(json!({
        "modalities": ["text"],
        "input_audio_format": qwen_audio_format_for_session(&config.input_audio_format),
        "sample_rate": config.sample_rate_hz,
        "input_audio_transcription": transcription,
        "turn_detection": null,
    }))
}

pub fn qwen_session_update_event_for_tts(config: &QwenTtsRealtimeConfig) -> QwenClientEvent {
    qwen_session_update_event(json!({
        "mode": "commit",
        "voice": config.voice,
        "language_type": "Auto",
        "response_format": qwen_audio_format_for_session(&config.output_audio_format),
        "sample_rate": config.sample_rate_hz,
    }))
}

pub fn qwen_session_update_event(session: Value) -> QwenClientEvent {
    QwenClientEvent {
        event_type: "session.update".to_string(),
        payload: json!({ "session": session }),
    }
}

pub fn qwen_audio_append_event(frame: &AudioInputFrame) -> QwenClientEvent {
    QwenClientEvent {
        event_type: "input_audio_buffer.append".to_string(),
        payload: json!({
            "audio": base64::engine::general_purpose::STANDARD.encode(samples_to_pcm16_le(&frame.samples)),
        }),
    }
}

pub fn qwen_audio_commit_event() -> QwenClientEvent {
    QwenClientEvent {
        event_type: "input_audio_buffer.commit".to_string(),
        payload: json!({}),
    }
}

pub fn qwen_input_text_append_event(text: impl Into<String>) -> QwenClientEvent {
    QwenClientEvent {
        event_type: "input_text_buffer.append".to_string(),
        payload: json!({ "text": text.into() }),
    }
}

pub fn qwen_input_text_commit_event() -> QwenClientEvent {
    QwenClientEvent {
        event_type: "input_text_buffer.commit".to_string(),
        payload: json!({}),
    }
}

pub fn qwen_session_finish_event() -> QwenClientEvent {
    QwenClientEvent {
        event_type: "session.finish".to_string(),
        payload: json!({}),
    }
}

pub fn qwen_response_create_event() -> QwenClientEvent {
    QwenClientEvent {
        event_type: "response.create".to_string(),
        payload: json!({}),
    }
}

pub fn qwen_server_event_to_output(
    event: &Value,
    sample_rate_hz: u32,
) -> Result<Option<QwenRealtimeServerOutput>, QwenRealtimeMappingError> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .ok_or(QwenRealtimeMappingError::MissingField("type"))?;

    match event_type {
        "conversation.item.input_audio_transcription.delta"
        | "conversation.item.input_audio_transcription.text"
        | "response.audio_transcript.delta"
        | "response.text.delta" => {
            let text = event
                .get("delta")
                .or_else(|| event.get("text"))
                .and_then(Value::as_str)
                .ok_or(QwenRealtimeMappingError::MissingField("delta"))?;
            Ok(Some(QwenRealtimeServerOutput::TranscriptFragment {
                text: text.to_string(),
                final_fragment: false,
            }))
        }
        "conversation.item.input_audio_transcription.completed"
        | "response.audio_transcript.done"
        | "response.text.done" => {
            let text = event
                .get("transcript")
                .or_else(|| event.get("text"))
                .or_else(|| event.get("delta"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(Some(QwenRealtimeServerOutput::TranscriptFragment {
                text: text.to_string(),
                final_fragment: true,
            }))
        }
        "response.audio.delta" => {
            let audio = event
                .get("delta")
                .or_else(|| event.get("audio"))
                .and_then(Value::as_str)
                .ok_or(QwenRealtimeMappingError::MissingField("delta"))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(audio)
                .map_err(|_| QwenRealtimeMappingError::InvalidAudio)?;
            Ok(Some(QwenRealtimeServerOutput::Audio(AudioOutputFrame {
                sequence: 0,
                sample_rate_hz,
                channels: 1,
                samples: pcm16_le_to_samples(&bytes),
                is_filler: false,
            })))
        }
        "response.done" | "session.finished" => {
            if event_type == "session.finished" {
                if let Some(text) = event.get("transcript").and_then(Value::as_str) {
                    if !text.trim().is_empty() {
                        return Ok(Some(QwenRealtimeServerOutput::TranscriptFragment {
                            text: text.to_string(),
                            final_fragment: true,
                        }));
                    }
                }
            }
            Ok(Some(QwenRealtimeServerOutput::Completed {
                reason: event
                    .get("reason")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| Some(event_type.to_string())),
            }))
        }
        "error" => Ok(Some(QwenRealtimeServerOutput::Error {
            code: event
                .get("code")
                .or_else(|| event.pointer("/error/code"))
                .and_then(Value::as_str)
                .map(str::to_string),
            message: event
                .get("message")
                .or_else(|| event.pointer("/error/message"))
                .or_else(|| event.pointer("/error/msg"))
                .and_then(Value::as_str)
                .unwrap_or("Qwen realtime API returned an error")
                .to_string(),
        })),
        _ => Ok(None),
    }
}

fn qwen_audio_format_for_session(format: &str) -> String {
    match format {
        "pcm16" => "pcm".to_string(),
        other => other.to_string(),
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

    #[test]
    fn websocket_url_targets_dashscope_realtime() {
        let cfg = QwenTtsRealtimeConfig {
            api_base: "https://example.test/".to_string(),
            model: "qwen3-tts-flash-realtime".to_string(),
            ..QwenTtsRealtimeConfig::default()
        };

        assert_eq!(
            cfg.websocket_url(),
            "wss://example.test/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"
        );
    }

    #[test]
    fn audio_append_event_encodes_pcm16_base64() {
        let event = qwen_audio_append_event(&AudioInputFrame {
            sequence: 1,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0, 1.0, -1.0],
        });

        assert_eq!(event.event_type, "input_audio_buffer.append");
        assert_eq!(event.payload["audio"], "AAD/fwCA");
    }

    #[test]
    fn server_event_maps_audio_delta() {
        let event = json!({
            "type": "response.audio.delta",
            "delta": "AQD//w=="
        });

        let output = qwen_server_event_to_output(&event, 24_000).unwrap();
        let Some(QwenRealtimeServerOutput::Audio(frame)) = output else {
            panic!("expected audio output");
        };
        assert_eq!(frame.sample_rate_hz, 24_000);
        assert_eq!(frame.samples.len(), 2);
    }
}
