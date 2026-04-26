use crate::types::{AuditEvent, AuditEventKind, SkillCall, SkillContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillOutput {
    pub spoken_summary: String,
    pub structured_payload: Option<Value>,
    pub audit_payload: Option<Value>,
}

#[derive(Debug, Error)]
pub enum SkillError {
    #[error("skill not found: {0}")]
    NotFound(String),
    #[error("skill execution failed: {0}")]
    Execution(String),
    #[error("skill rejected by policy: {0}")]
    Rejected(String),
    #[error("skill argument schema validation failed: {0}")]
    SchemaViolation(String),
}

#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> Value;
    async fn execute(&self, args: Value) -> Result<SkillOutput, SkillError>;
}

#[async_trait]
pub trait SkillExecutor: Send + Sync {
    async fn execute(
        &self,
        call: SkillCall,
        context: SkillContext,
    ) -> Result<SkillOutput, SkillError>;
}

// ---------------------------------------------------------------------------
// AuditSink (Phase 2)
// ---------------------------------------------------------------------------

/// Receives audit events from the skill execution pipeline.
/// The default implementation is a no-op; replace with a real sink for
/// observability (e.g., tracing spans, append-only log, metrics counter).
#[async_trait]
pub trait AuditSink: Send + Sync {
    async fn record(&self, event: AuditEvent);
}

pub struct NoOpAuditSink;

#[async_trait]
impl AuditSink for NoOpAuditSink {
    async fn record(&self, _event: AuditEvent) {}
}

// ---------------------------------------------------------------------------
// SkillRegistry
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SkillRegistry {
    skills: BTreeMap<String, Arc<dyn Skill>>,
    audit_sink: Arc<dyn AuditSink>,
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self {
            skills: BTreeMap::new(),
            audit_sink: Arc::new(NoOpAuditSink),
        }
    }
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_audit_sink(mut self, sink: Arc<dyn AuditSink>) -> Self {
        self.audit_sink = sink;
        self
    }

    pub fn register<S>(&mut self, skill: S)
    where
        S: Skill + 'static,
    {
        self.skills
            .insert(skill.name().to_string(), Arc::new(skill));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Skill>> {
        self.skills.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn generate_tool_prompt(&self) -> String {
        self.skills
            .values()
            .map(|skill| {
                format!(
                    "- {}: {} schema={} ",
                    skill.name(),
                    skill.description(),
                    skill.schema()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Execute a skill call with schema validation, audit events, and error propagation.
    /// Skills whose `schema()` returns `Value::Null` skip validation.
    pub async fn execute_audited(
        &self,
        call: SkillCall,
        session_id: &str,
    ) -> Result<SkillOutput, SkillError> {
        let skill = self
            .skills
            .get(&call.name)
            .cloned()
            .ok_or_else(|| SkillError::NotFound(call.name.clone()))?;

        // Schema validation gate
        let schema = skill.schema();
        if schema != Value::Null {
            let compiled = jsonschema::validator_for(&schema)
                .map_err(|e| SkillError::SchemaViolation(e.to_string()))?;
            if let Err(err) = compiled.validate(&call.arguments) {
                let reason = err.to_string();
                self.audit_sink
                    .record(AuditEvent::now(
                        session_id,
                        AuditEventKind::SchemaViolation {
                            name: call.name.clone(),
                            reason: reason.clone(),
                        },
                    ))
                    .await;
                return Err(SkillError::SchemaViolation(reason));
            }
        }

        let args_summary = truncate_json_summary(&call.arguments, 200);
        self.audit_sink
            .record(AuditEvent::now(
                session_id,
                AuditEventKind::SkillAttempt {
                    name: call.name.clone(),
                    args_summary,
                },
            ))
            .await;

        let start = std::time::Instant::now();
        let result = skill.execute(call.arguments.clone()).await;
        let duration_ms = start.elapsed().as_millis() as u64;
        let success = result.is_ok();

        self.audit_sink
            .record(AuditEvent::now(
                session_id,
                AuditEventKind::SkillResult {
                    name: call.name.clone(),
                    success,
                    duration_ms,
                },
            ))
            .await;

        result
    }
}

#[async_trait]
impl SkillExecutor for SkillRegistry {
    async fn execute(
        &self,
        call: SkillCall,
        context: SkillContext,
    ) -> Result<SkillOutput, SkillError> {
        self.execute_audited(call, &context.session_id).await
    }
}

fn truncate_json_summary(value: &Value, max_chars: usize) -> String {
    let s = value.to_string();
    if s.len() <= max_chars {
        s
    } else {
        format!("{}…", &s[..max_chars])
    }
}

