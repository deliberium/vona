use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioInputFrame {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioOutputFrame {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
    pub is_filler: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFrameEncoding {
    PcmS16Le,
    PcmF32Le,
    Opus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedAudioFrame {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub encoding: AudioFrameEncoding,
    pub bytes: Vec<u8>,
    pub is_filler: bool,
}

impl EncodedAudioFrame {
    pub fn pcm_s16le_from_output(frame: &AudioOutputFrame) -> Self {
        let mut bytes = Vec::with_capacity(frame.samples.len() * 2);
        for sample in &frame.samples {
            let sample = sample.clamp(-1.0, 1.0);
            let pcm = if sample < 0.0 {
                (sample * 32768.0).round() as i16
            } else {
                (sample * 32767.0).round() as i16
            };
            bytes.extend_from_slice(&pcm.to_le_bytes());
        }
        Self {
            sequence: frame.sequence,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            encoding: AudioFrameEncoding::PcmS16Le,
            bytes,
            is_filler: frame.is_filler,
        }
    }

    pub fn decode_pcm_s16le_samples(&self) -> Option<Vec<f32>> {
        if self.encoding != AudioFrameEncoding::PcmS16Le || !self.bytes.len().is_multiple_of(2) {
            return None;
        }
        Some(
            self.bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
                .collect(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillContext {
    pub session_id: String,
    pub user_id: Option<String>,
    pub thread_id: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlEvent {
    SkillCall(SkillCall),
    TranscriptFragment { text: String, final_fragment: bool },
    Interruption { reason: Option<String> },
    Diagnostic { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalContextEvent {
    pub source: String,
    pub spoken_summary: Option<String>,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub time_to_first_audio_ms: Option<u64>,
    pub interruptions: u64,
    pub tool_calls: u64,
    pub fallback_count: u64,
}

// ---------------------------------------------------------------------------
// Audit types (Phase 2)
// ---------------------------------------------------------------------------

/// A single auditable event emitted by the skill execution pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub session_id: String,
    pub kind: AuditEventKind,
    /// Milliseconds since Unix epoch at the time of emission.
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventKind {
    SkillAttempt {
        name: String,
        /// Truncated JSON summary of arguments (not the full payload).
        args_summary: String,
    },
    SkillResult {
        name: String,
        success: bool,
        duration_ms: u64,
    },
    SchemaViolation {
        name: String,
        reason: String,
    },
    ToolTimeout {
        name: String,
        budget_ms: u64,
    },
    Fallback {
        reason: String,
    },
}

impl AuditEvent {
    /// Convenience constructor using the current system time.
    pub fn now(session_id: impl Into<String>, kind: AuditEventKind) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            session_id: session_id.into(),
            kind,
            timestamp_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioFrameEncoding, AudioOutputFrame, EncodedAudioFrame};

    #[test]
    fn encoded_audio_frame_round_trips_pcm_s16le_samples() {
        let frame = AudioOutputFrame {
            sequence: 7,
            sample_rate_hz: 24_000,
            channels: 1,
            samples: vec![0.0, 0.5, -0.5],
            is_filler: false,
        };

        let encoded = EncodedAudioFrame::pcm_s16le_from_output(&frame);
        assert_eq!(encoded.sequence, 7);
        assert_eq!(encoded.sample_rate_hz, 24_000);
        assert_eq!(encoded.encoding, AudioFrameEncoding::PcmS16Le);
        assert_eq!(encoded.bytes.len(), 6);

        let decoded = encoded.decode_pcm_s16le_samples().unwrap();
        assert_eq!(decoded.len(), frame.samples.len());
        for (left, right) in decoded.iter().zip(frame.samples.iter()) {
            assert!((left - right).abs() < 0.0001);
        }
    }
}
