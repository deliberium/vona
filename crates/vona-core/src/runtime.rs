use crate::skills::{SkillError, SkillExecutor};
use crate::types::{AuditEvent, AuditEventKind, ControlEvent, ExternalContextEvent, SkillContext};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{Duration, timeout};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    BackendUnavailable,
    ControlRejected,
    ToolTimeout,
    ToolFailed,
    Silence,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillerStrategy {
    None,
    StaticClip,
    BackendGenerated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimeDecision {
    Ignore,
    Continue,
    InjectContext(ExternalContextEvent),
    Fallback { reason: FallbackReason },
}

pub trait SessionPolicy: Send + Sync {
    fn should_accept_control_event(&self, event: &ControlEvent) -> bool;
    fn should_fallback_to_bridge(&self, reason: &FallbackReason) -> bool;
    fn max_tool_latency_ms(&self) -> u64;
}

pub struct VonaRuntime<E: SkillExecutor + ?Sized, P: SessionPolicy + ?Sized> {
    skill_executor: Arc<E>,
    policy: Arc<P>,
    filler_strategy: FillerStrategy,
}

impl<E, P> VonaRuntime<E, P>
where
    E: SkillExecutor + ?Sized,
    P: SessionPolicy + ?Sized,
{
    pub fn new(skill_executor: Arc<E>, policy: Arc<P>, filler_strategy: FillerStrategy) -> Self {
        Self {
            skill_executor,
            policy,
            filler_strategy,
        }
    }

    pub fn filler_strategy(&self) -> &FillerStrategy {
        &self.filler_strategy
    }

    pub async fn handle_control_event(
        &self,
        event: &ControlEvent,
        context: SkillContext,
    ) -> Result<RuntimeDecision, SkillError> {
        if !self.policy.should_accept_control_event(event) {
            return Ok(
                if self
                    .policy
                    .should_fallback_to_bridge(&FallbackReason::ControlRejected)
                {
                    RuntimeDecision::Fallback {
                        reason: FallbackReason::ControlRejected,
                    }
                } else {
                    RuntimeDecision::Ignore
                },
            );
        }

        match event {
            ControlEvent::SkillCall(call) => {
                let budget = Duration::from_millis(self.policy.max_tool_latency_ms());
                let result = timeout(
                    budget,
                    self.skill_executor.execute(call.clone(), context.clone()),
                )
                .await;

                match result {
                    Ok(Ok(output)) => Ok(RuntimeDecision::InjectContext(ExternalContextEvent {
                        source: format!("skill:{}", call.name),
                        spoken_summary: Some(output.spoken_summary),
                        payload: output.structured_payload.unwrap_or(serde_json::Value::Null),
                    })),
                    Ok(Err(err)) => {
                        if self
                            .policy
                            .should_fallback_to_bridge(&FallbackReason::ToolFailed)
                        {
                            Ok(RuntimeDecision::Fallback {
                                reason: FallbackReason::ToolFailed,
                            })
                        } else {
                            Err(err)
                        }
                    }
                    Err(_elapsed) => {
                        // Timeout — emit fallback decision; caller can record the AuditEvent
                        // if it has access to a session_id and an AuditSink.
                        if self
                            .policy
                            .should_fallback_to_bridge(&FallbackReason::ToolTimeout)
                        {
                            Ok(RuntimeDecision::Fallback {
                                reason: FallbackReason::ToolTimeout,
                            })
                        } else {
                            Err(SkillError::Execution(format!(
                                "tool '{}' exceeded latency budget of {} ms",
                                call.name,
                                self.policy.max_tool_latency_ms()
                            )))
                        }
                    }
                }
            }
            ControlEvent::TranscriptFragment { .. }
            | ControlEvent::Interruption { .. }
            | ControlEvent::Diagnostic { .. } => Ok(RuntimeDecision::Continue),
        }
    }

    /// Variant of `handle_control_event` that also records audit events for timeouts.
    pub async fn handle_control_event_audited(
        &self,
        event: &ControlEvent,
        context: SkillContext,
        audit_sink: &(impl crate::skills::AuditSink + ?Sized),
    ) -> Result<RuntimeDecision, SkillError> {
        let decision = self.handle_control_event(event, context.clone()).await?;

        if let RuntimeDecision::Fallback {
            reason: FallbackReason::ToolTimeout,
        } = &decision
            && let ControlEvent::SkillCall(call) = event
        {
            audit_sink
                .record(AuditEvent::now(
                    &context.session_id,
                    AuditEventKind::ToolTimeout {
                        name: call.name.clone(),
                        budget_ms: self.policy.max_tool_latency_ms(),
                    },
                ))
                .await;
        }

        Ok(decision)
    }
}
