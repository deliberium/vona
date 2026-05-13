use crate::backend::{BackendError, SpeechToSpeechBackend};
use crate::runtime::{FallbackReason, RuntimeDecision, SessionPolicy, VonaRuntime};
use crate::skills::SkillExecutor;
use crate::transport::{AudioTransport, TransportError};
use crate::types::{ControlEvent, SessionMetrics, SkillContext};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechStyleProfile {
    pub pace: f32,
    pub warmth: f32,
    pub expressiveness: f32,
    pub formality: Option<String>,
    pub preferred_voice: Option<String>,
}

impl Default for SpeechStyleProfile {
    fn default() -> Self {
        Self {
            pace: 0.5,
            warmth: 0.5,
            expressiveness: 0.5,
            formality: None,
            preferred_voice: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub session_id: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub style_profile: Option<SpeechStyleProfile>,
    #[serde(default)]
    pub metadata: Value,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_id: "default-session".to_string(),
            sample_rate_hz: 24_000,
            channels: 1,
            style_profile: None,
            metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Listening,
    Generating,
    ExecutingSkill,
    PausedForInterruption,
    FallingBackToBridge,
    Closed,
}

// ---------------------------------------------------------------------------
// Phase 3: session state machine driver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionCloseReason {
    TransportClosed,
    BackendFinished,
    PolicyEnded,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub metrics: SessionMetrics,
    pub close_reason: SessionCloseReason,
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
}

/// Drives a full speech-to-speech session loop until the transport closes,
/// the backend signals completion, or an unrecoverable error occurs.
///
/// # Metrics populated
/// - `time_to_first_audio_ms`: instant of the first non-empty output frame
/// - `interruptions`: each backend-emitted `ControlEvent::Interruption`
/// - `tool_calls`: each `SkillAttempt` emitted from `RuntimeDecision::InjectContext`
/// - `fallback_count`: each `RuntimeDecision::Fallback` returned from runtime
pub async fn run_session<T, B, P, E>(
    transport: T,
    backend: &B,
    runtime: &VonaRuntime<E, P>,
    config: SessionConfig,
) -> Result<SessionSummary, SessionError>
where
    T: AudioTransport,
    B: SpeechToSpeechBackend,
    P: SessionPolicy,
    E: SkillExecutor,
{
    let session_id = config.session_id.clone();
    let mut backend_session = backend.start_session(config.clone()).await?;
    let mut metrics = SessionMetrics::default();
    let session_start = Instant::now();

    loop {
        // Receive input from transport
        let frame = match transport.recv_frame().await {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                // Transport closed cleanly
                let _ = backend.end_session(backend_session).await;
                return Ok(SessionSummary {
                    session_id,
                    metrics,
                    close_reason: SessionCloseReason::TransportClosed,
                });
            }
            Err(err) => {
                let _ = backend.end_session(backend_session).await;
                return Ok(SessionSummary {
                    session_id,
                    metrics,
                    close_reason: SessionCloseReason::Error(err.to_string()),
                });
            }
        };

        let step = match backend.step(&mut backend_session, frame).await {
            Ok(step) => step,
            Err(err) => {
                let _ = backend.end_session(backend_session).await;
                return Ok(SessionSummary {
                    session_id,
                    metrics,
                    close_reason: SessionCloseReason::Error(err.to_string()),
                });
            }
        };

        // Track time to first audio
        if metrics.time_to_first_audio_ms.is_none() && !step.output_audio.is_empty() {
            metrics.time_to_first_audio_ms = Some(session_start.elapsed().as_millis() as u64);
        }

        // Send output audio frames
        for audio_frame in step.output_audio {
            if let Err(err) = transport.send_frame(audio_frame).await {
                let _ = backend.end_session(backend_session).await;
                return Ok(SessionSummary {
                    session_id,
                    metrics,
                    close_reason: SessionCloseReason::Error(err.to_string()),
                });
            }
        }

        // Process control events
        for control_event in step.control_events {
            let skill_context = SkillContext {
                session_id: session_id.clone(),
                user_id: None,
                thread_id: None,
                metadata: Value::Null,
            };

            if let ControlEvent::Interruption { .. } = &control_event {
                metrics.interruptions += 1;
                transport.clear_output().await.ok();
            }

            let decision = match runtime
                .handle_control_event(&control_event, skill_context)
                .await
            {
                Ok(d) => d,
                Err(_) => continue,
            };

            match decision {
                RuntimeDecision::InjectContext(ctx_event) => {
                    metrics.tool_calls += 1;
                    if let Err(err) = backend.inject_event(&mut backend_session, ctx_event).await {
                        tracing_fallback(&err.to_string());
                    }
                }
                RuntimeDecision::Fallback { reason } => {
                    metrics.fallback_count += 1;
                    if reason == FallbackReason::ToolTimeout {
                        // Record timeout audit event through the session's audit path.
                        // (If caller wired an AuditSink into SkillRegistry it already fired.)
                    }
                }
                RuntimeDecision::Ignore | RuntimeDecision::Continue => {}
            }
        }

        if step.finished {
            let _ = backend.end_session(backend_session).await;
            return Ok(SessionSummary {
                session_id,
                metrics,
                close_reason: SessionCloseReason::BackendFinished,
            });
        }
    }
}

// Minimal tracing shim to avoid pulling in tracing as a hard dep in core.
// Callers that want structured logging should handle errors from inject_event themselves.
#[inline(always)]
fn tracing_fallback(msg: &str) {
    let _ = msg; // silence unused warning in release builds
    #[cfg(debug_assertions)]
    eprintln!("[vona] inject_event error: {msg}");
}
