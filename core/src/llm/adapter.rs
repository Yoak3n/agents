use async_trait::async_trait;

use crate::schema::common::{Message, ModelProvider, ToolCall, ToolDefinition};
use crate::schema::common::EventListener;

use crate::error::AgentError;

/// LLM adapter 的抽象接口，不同 API 格式各自实现
#[async_trait]
pub trait LlmAdapter: Send + Sync {
    async fn chat(
        &self,
        provider: &ModelProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        listener: &dyn EventListener,
    ) -> Result<AgentResponse, AgentError>;
}

/// adapter 返回值，决定 agent loop 是否继续
#[derive(Debug)]
pub enum AgentResponse {
    /// LLM 返回了最终文本回复（无工具调用）
    MessageComplete(Message),
    /// LLM 请求调用工具（可能多个并行）
    ToolCalls(Vec<ToolCall>),
}
