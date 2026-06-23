use std::sync::Arc;

use async_trait::async_trait;

use yoakore::estimate_tokens;
use yoakore::prelude::*;

/// Context management strategy — defines how the message list is compressed
/// when it approaches the context window limit.
///
/// Implement this trait to replace the default sliding-window strategy.
#[async_trait]
pub trait ContextManager: Send + Sync {
    /// Check whether the message list exceeds the context window and compress if needed.
    async fn manage(&self, messages: &mut Vec<Message>, max_tokens: u32);
}

/// Configuration for the default context manager.
pub struct ContextConfig {
    /// Number of recent messages to always keep (never replaced by a summary).
    pub recent_to_keep: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { recent_to_keep: 10 }
    }
}

/// Default context manager — sliding-window strategy.
///
/// When token usage exceeds 80% of the context window, keeps system messages
/// plus the N most recent messages and replaces the middle with a summary.
pub struct DefaultContext {
    recent_to_keep: usize,
}

impl DefaultContext {
    /// Create a new instance from the given config.
    pub fn new(config: ContextConfig) -> Self {
        Self {
            recent_to_keep: config.recent_to_keep,
        }
    }
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

#[async_trait]
impl ContextManager for DefaultContext {
    async fn manage(&self, messages: &mut Vec<Message>, max_tokens: u32) {
        let total: u32 = messages
            .iter()
            .map(|m| estimate_tokens(&m.content) as u32)
            .sum();
        let threshold = (max_tokens as f64 * 0.8) as u32;

        if total <= threshold {
            return;
        }

        log::info!(
            "DefaultContext: {} tokens exceeds threshold {} (max {}), summarizing",
            total,
            threshold,
            max_tokens
        );

        let keep = self.recent_to_keep;
        if messages.len() <= keep + 1 {
            return;
        }

        let first_non_system = messages
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(0);
        let split_point = messages.len() - keep;

        if split_point <= first_non_system {
            return;
        }

        let to_summarize: Vec<String> = messages[first_non_system..split_point]
            .iter()
            .map(|m| format!("[{}]: {}", role_str(&m.role), m.content))
            .collect();

        let summary = format!(
            "[Context summary of {} earlier messages]\n{}",
            to_summarize.len(),
            to_summarize.join("\n")
        );

        let summary_msg = Message::system(summary);
        messages.splice(first_non_system..split_point, std::iter::once(summary_msg));

        let new_total: u32 = messages
            .iter()
            .map(|m| estimate_tokens(&m.content) as u32)
            .sum();
        log::info!(
            "DefaultContext: summarized {} messages, now {} tokens",
            split_point - first_non_system,
            new_total
        );
    }
}

/// Bridges [`ContextManager`] into an [`AgentHook`].
pub struct ContextHook {
    manager: Arc<dyn ContextManager>,
    max_tokens: u32,
}

impl ContextHook {
    /// Create a new hook from the given manager and token limit.
    pub fn new(manager: Arc<dyn ContextManager>, max_tokens: u32) -> Self {
        Self {
            manager,
            max_tokens,
        }
    }
}

#[async_trait]
impl AgentHook for ContextHook {
    async fn before_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        messages: &mut Vec<Message>,
    ) -> HookResult {
        self.manager.manage(messages, self.max_tokens).await;
        HookResult::Continue
    }
}
