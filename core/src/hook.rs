use async_trait::async_trait;

use crate::schema::common::{Message, ModelProvider, ToolCall};

use crate::llm::AgentResponse;

/// Hook 执行结果
#[derive(Debug)]
pub enum HookResult {
    /// 正常继续
    Continue,
    /// 跳过本次操作（如跳过某个工具调用）
    Skip,
    /// 中止循环，返回错误信息
    Abort(String),
}

/// Hook 上下文，携带当前 agent loop 的运行时信息
pub struct HookContext<'a> {
    pub provider: &'a ModelProvider,
    pub round: usize,
    pub session_id: &'a str,
}

/// Agent hook trait，所有方法都有默认空实现，用户按需覆盖
#[async_trait]
pub trait AgentHook: Send + Sync {
    /// LLM 调用前，可修改消息列表
    async fn before_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        _messages: &mut Vec<Message>,
    ) -> HookResult {
        HookResult::Continue
    }

    /// LLM 调用后，可检查/替换响应
    async fn after_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        _response: &mut AgentResponse,
    ) -> HookResult {
        HookResult::Continue
    }

    /// 工具执行前
    async fn before_tool_call(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
    ) -> HookResult {
        HookResult::Continue
    }

    /// 工具执行后，可修改结果
    async fn after_tool_call(
        &self,
        _ctx: &HookContext<'_>,
        _call: &ToolCall,
        _result: &mut String,
    ) -> HookResult {
        HookResult::Continue
    }

    /// LLM 流式输出每收到一个 delta chunk 时触发
    async fn on_llm_delta(&self, _ctx: &HookContext<'_>, _delta: &str) {}

    /// 发生错误时
    async fn on_error(&self, _ctx: &HookContext<'_>, _error: &str) {}
}
