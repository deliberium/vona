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
