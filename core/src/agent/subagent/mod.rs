pub mod registry;

use crate::schema::common::{AppConfig, Message};
#[cfg(feature = "extension")]
use crate::schema::extension::skill::Skill;
use async_trait::async_trait;

/// Sub-agent unified interface.
#[async_trait]
pub trait SubAgent: Send + Sync {
    /// Unique name (used as registry key).
    fn name(&self) -> &str;

    /// Human-readable capability description.
    fn description(&self) -> &str;

    /// Capability tags for introspection and routing (e.g. ["search", "summarize"]).
    fn capabilities(&self) -> Vec<String> {
        vec![]
    }

    /// Execute the sub-agent task.
    async fn execute(&self, input: &str, context: SubAgentContext<'_>) -> SubAgentResult;
}

/// Context passed to a sub-agent during execution.
pub struct SubAgentContext<'a> {
    pub session_id: &'a str,
    pub message_history: &'a [Message],
    /// Parent registry for hierarchical composition. Sub-agents can use this
    /// to delegate work to other sub-agents.
    pub registry: Option<&'a super::subagent::SubAgentRegistry>,
    #[cfg(feature = "extension")]
    pub available_skills: &'a [Skill],
    pub app_config: &'a AppConfig,
}

/// Sub-agent execution result.
pub struct SubAgentResult {
    /// Primary text output.
    pub output: String,
    /// Execution status.
    pub status: SubAgentStatus,
    /// Optional structured metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Execution status of a sub-agent result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SubAgentStatus {
    /// Completed successfully.
    #[default]
    Success,
    /// Partial result — some work done but incomplete.
    Partial,
    /// Failed with an error message.
    Failed(String),
}

impl SubAgentResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            status: SubAgentStatus::Success,
            metadata: None,
        }
    }

    pub fn partial(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            status: SubAgentStatus::Partial,
            metadata: None,
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            output: String::new(),
            status: SubAgentStatus::Failed(error.into()),
            metadata: None,
        }
    }
}

pub use registry::{SubAgentError, SubAgentRegistry};
