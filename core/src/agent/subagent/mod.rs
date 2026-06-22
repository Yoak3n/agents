pub mod registry;

use async_trait::async_trait;
use crate::schema::common::{AppConfig, Message};
#[cfg(feature = "extension")]
use crate::schema::extension::skill::Skill;
/// Sub-agent unified interface.
#[async_trait]
pub trait SubAgent: Send + Sync {
    /// Unique name (used as registry key).
    fn name(&self) -> &str;

    /// Human-readable capability description.
    fn description(&self) -> &str;

    /// Execute the sub-agent task.
    async fn execute(&self, input: &str, context: SubAgentContext<'_>) -> SubAgentResult;
}

/// Context passed to a sub-agent during execution.
pub struct SubAgentContext<'a> {
    pub session_id: &'a str,
    pub message_history: &'a [Message],
    #[cfg(feature = "extension")]
    pub available_skills: &'a [Skill],
    pub app_config: &'a AppConfig,
}

/// Sub-agent execution result.
pub struct SubAgentResult {
    /// Primary text output.
    pub output: String,
    /// Optional structured metadata.
    pub metadata: Option<serde_json::Value>,
}

pub use registry::{SubAgentRegistry, SubAgentError};
