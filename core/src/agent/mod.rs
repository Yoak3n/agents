pub mod base;
pub mod builder;
pub mod plan;
pub mod subagent;
pub mod team;

use std::sync::Arc;

use crate::error::AgentError;
use crate::llm::{AgentResponse, LlmAdapter};
use crate::schema::common::{EventListener, Message, ModelProvider, ToolDefinition};

pub use base::BaseAgent;
pub use team::{TeamAgent, TeamResult};

/// Common trait for any entity that can interact with the LLM.
#[async_trait::async_trait]
pub trait AgentLike: Send + Sync {
    /// Run a single LLM turn (one request/response cycle).
    async fn run_turn(
        &self,
        provider: &ModelProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        listener: &dyn EventListener,
    ) -> Result<AgentResponse, AgentError>;

    /// Get the max tool rounds allowed.
    fn max_tool_rounds(&self) -> usize;

    /// 最低模型等级要求。provider.tier >= min_tier 才允许使用。
    fn min_tier(&self) -> u8 {
        1
    }

    /// Get a reference to the underlying LLM adapter.
    fn adapter(&self) -> Arc<dyn LlmAdapter>;
}

/// Abstraction for tool execution, provided by Runtime to sub-agents.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool by name with given arguments.
    async fn call(&self, name: &str, args: serde_json::Value) -> Result<String, String>;

    /// Get definitions of all available tools.
    fn definitions(&self) -> Vec<ToolDefinition>;
}
