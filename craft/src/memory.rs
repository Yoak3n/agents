use std::sync::Arc;

use async_trait::async_trait;

use yoakore::prelude::*;
use yoakore::{AgentResponseKind, MemoryEntry, Storage};

/// Memory provider trait — defines how memories are retrieved and extracted.
///
/// Implement this trait to fully replace the memory system behavior.
pub trait MemoryProvider: Send + Sync {
    /// Retrieve memories relevant to the current conversation.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<MemoryEntry>> + Send + 'a>>;

    /// Extract a new memory worth saving from an assistant reply.
    /// Returns `None` if there is nothing worth remembering.
    fn extract<'a>(
        &'a self,
        assistant_reply: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<NewMemory>> + Send + 'a>>;
}

/// A new memory extracted from an assistant reply.
pub struct NewMemory {
    pub title: String,
    pub content: String,
    pub tags: Option<String>,
}

/// Default memory implementation backed by [`Storage`].
///
/// Uses LIKE-based keyword search and a simple content-length extraction strategy.
pub struct DefaultMemory {
    storage: Arc<Storage>,
    auto_extract: bool,
}

impl DefaultMemory {
    /// Create a new instance with the given storage backend.
    pub fn new(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            auto_extract: true,
        }
    }

    /// Enable or disable automatic memory extraction from assistant replies.
    pub fn auto_extract(mut self, enabled: bool) -> Self {
        self.auto_extract = enabled;
        self
    }
}

impl MemoryProvider for DefaultMemory {
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<MemoryEntry>> + Send + 'a>> {
        Box::pin(async move {
            self.storage
                .search_memories(query, 0, limit as i64)
                .unwrap_or_default()
        })
    }

    fn extract<'a>(
        &'a self,
        assistant_reply: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<NewMemory>> + Send + 'a>> {
        Box::pin(async move {
            if !self.auto_extract || assistant_reply.len() < 50 {
                return None;
            }
            Some(NewMemory {
                title: assistant_reply[..assistant_reply.len().min(100)].to_string(),
                content: assistant_reply.to_string(),
                tags: Some("auto-extracted".to_string()),
            })
        })
    }
}

/// Memory system configuration.
pub struct MemoryConfig {
    /// Maximum number of relevant memories to inject per LLM call.
    pub max_injected: usize,
    /// Whether to automatically extract memories from assistant replies.
    pub auto_extract: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_injected: 5,
            auto_extract: true,
        }
    }
}

/// Bridges [`MemoryProvider`] into an [`AgentHook`].
///
/// Before each LLM call, retrieves relevant memories and injects them as a system message.
/// After each LLM call, extracts new memories from the assistant reply.
pub struct MemoryHook {
    provider: Arc<dyn MemoryProvider>,
    max_injected: usize,
    storage: Arc<Storage>,
}

impl MemoryHook {
    /// Create a new hook with the given provider, storage, and injection limit.
    pub fn new(
        provider: Arc<dyn MemoryProvider>,
        storage: Arc<Storage>,
        max_injected: usize,
    ) -> Self {
        Self {
            provider,
            max_injected,
            storage,
        }
    }
}

#[async_trait]
impl AgentHook for MemoryHook {
    async fn before_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        messages: &mut Vec<Message>,
    ) -> HookResult {
        let query = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        if query.is_empty() {
            return HookResult::Continue;
        }

        match self.provider.retrieve(query, self.max_injected).await {
            memories if !memories.is_empty() => {
                let mem_text: Vec<String> = memories
                    .iter()
                    .map(|m| format!("- {} (weight: {:.2})", m.content, m.weight))
                    .collect();
                let injection = format!(
                    "<relevant-memories>\n{}\n</relevant-memories>",
                    mem_text.join("\n")
                );
                if let Some(pos) = messages.iter().position(|m| m.role == Role::User) {
                    messages.insert(pos, Message::system(injection));
                }
            }
            _ => {}
        }

        HookResult::Continue
    }

    async fn after_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        response: &mut AgentResponse,
    ) -> HookResult {
        let content = match &response.kind {
            AgentResponseKind::MessageComplete(msg) => &msg.content,
            _ => return HookResult::Continue,
        };

        if let Some(new_mem) = self.provider.extract(content).await
            && let Err(e) = self.storage.save_memory(
                None,
                &new_mem.title,
                &new_mem.content,
                new_mem.tags.as_deref(),
                None,
                None,
            )
        {
            log::warn!("MemoryHook: failed to save memory: {}", e);
        }

        HookResult::Continue
    }
}
