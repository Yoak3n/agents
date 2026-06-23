use async_trait::async_trait;

use crate::schema::common::EventListener;
use crate::schema::common::{Message, ModelProvider, ToolCall, ToolDefinition};

use crate::error::AgentError;

/// LLM API 返回的 token 用量信息。
///
/// 当 adapter 支持解析 `usage` 字段时填充，否则为 `None`。
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// 缓存命中的 input tokens（OpenAI: `prompt_tokens_details.cached_tokens`）
    pub cached_input_tokens: u64,
}

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

/// adapter 返回值，包含响应内容和 token 用量。
#[derive(Debug)]
pub struct AgentResponse {
    pub kind: AgentResponseKind,
    pub usage: Option<Usage>,
}

impl AgentResponse {
    pub fn message_complete(msg: Message) -> Self {
        Self {
            kind: AgentResponseKind::MessageComplete(msg),
            usage: None,
        }
    }

    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            kind: AgentResponseKind::ToolCalls(calls),
            usage: None,
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// 响应内容类型
#[derive(Debug)]
pub enum AgentResponseKind {
    /// LLM 返回了最终文本回复（无工具调用）
    MessageComplete(Message),
    /// LLM 请求调用工具（可能多个并行）
    ToolCalls(Vec<ToolCall>),
}
