use std::sync::{Arc, Mutex};

use crate::schema::common::{AgentEvent, EventListener, Message, ModelProvider, NullListener, ToolDefinition};
use crate::llm::{AgentResponse, LlmAdapter};
use crate::error::AgentError;
use crate::agent::AgentLike;
use crate::hook::{AgentHook, HookContext, HookResult};
use crate::tools::{ToolRegistry, ProcessManager};

/// Wrapper that adapts a `FnMut(&AgentEvent)` closure into an `EventListener`.
struct FnEventListener<F: FnMut(&AgentEvent) + Send + Sync>(Mutex<F>);

impl<F: FnMut(&AgentEvent) + Send + Sync> EventListener for FnEventListener<F> {
    fn on_event(&self, event: &AgentEvent) {
        if let Ok(mut f) = self.0.lock() {
            f(event);
        }
    }
}

/// Base agent — the foundation for all agent types.
///
/// - `execute()` — simplest: just returns the final reply
/// - Use `with_listener()` to receive streaming events (Delta, ToolCallStart, etc.)
/// - Use `with_hooks()` for lifecycle interception (before/after LLM, before/after tool)
pub struct BaseAgent {
    adapter: Arc<dyn LlmAdapter>,
    max_tool_rounds: usize,
    min_tier: u8,
    tools: Option<Arc<ToolRegistry>>,
    hooks: Option<Arc<dyn AgentHook>>,
    listener: Arc<dyn EventListener>,
    process_manager: Arc<ProcessManager>,
}

impl BaseAgent {
    pub fn new(adapter: impl LlmAdapter + 'static) -> Self {
        Self {
            adapter: Arc::new(adapter),
            max_tool_rounds: 200,
            min_tier: 1,
            tools: None,
            hooks: None,
            listener: Arc::new(NullListener),
            process_manager: Arc::new(ProcessManager::new()),
        }
    }

    pub fn with_max_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = max;
        self
    }

    pub fn with_min_tier(mut self, tier: u8) -> Self {
        self.min_tier = tier;
        self
    }

    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_hooks(mut self, hooks: impl AgentHook + 'static) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    pub fn with_hooks_arc(mut self, hooks: Arc<dyn AgentHook>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    pub fn with_listener(mut self, listener: impl EventListener + 'static) -> Self {
        self.listener = Arc::new(listener);
        self
    }

    pub fn with_listener_arc(mut self, listener: Arc<dyn EventListener>) -> Self {
        self.listener = listener;
        self
    }

    /// Convenience: accept a closure for event callbacks.
    ///
    /// ```ignore
    /// agent.with_on_event(|event| {
    ///     if let AgentEvent::Delta(text) = event {
    ///         print!("{}", text);
    ///     }
    /// })
    /// ```
    pub fn with_on_event(mut self, f: impl FnMut(&AgentEvent) + Send + Sync + 'static) -> Self {
        self.listener = Arc::new(FnEventListener(Mutex::new(f)));
        self
    }

    pub fn with_process_manager(mut self, pm: Arc<ProcessManager>) -> Self {
        self.process_manager = pm;
        self
    }

    /// Run the full agent loop: LLM → tool execution → feed back → repeat.
    ///
    /// Returns the final assistant message content. Events are delivered through
    /// the `EventListener` set via `with_listener()`.
    pub async fn execute(
        &self,
        provider: &ModelProvider,
        messages: &mut Vec<Message>,
    ) -> Result<String, AgentError> {
        let tools = self.tool_definitions();
        let session_id = uuid::Uuid::new_v4().to_string();

        for round in 0..self.max_tool_rounds {
            let ctx = HookContext {
                provider,
                round,
                session_id: &session_id,
            };

            // ── before_llm_call hook ──
            if let Some(ref hooks) = self.hooks {
                match hooks.before_llm_call(&ctx, messages).await {
                    HookResult::Abort(reason) => {
                        self.emit(AgentEvent::Error(reason.clone()));
                        return Err(AgentError::Other(reason));
                    }
                    HookResult::Skip => continue,
                    HookResult::Continue => {}
                }
            }

            // ── LLM call ──
            self.emit(AgentEvent::Thinking);
            let response = self.adapter.chat(provider, messages, &tools, &*self.listener).await?;

            // ── after_llm_call hook ──
            let mut response = response;
            if let Some(ref hooks) = self.hooks {
                match hooks.after_llm_call(&ctx, &mut response).await {
                    HookResult::Abort(reason) => {
                        self.emit(AgentEvent::Error(reason.clone()));
                        return Err(AgentError::Other(reason));
                    }
                    HookResult::Skip => continue,
                    HookResult::Continue => {}
                }
            }

            match response {
                AgentResponse::MessageComplete(msg) => {
                    let content = msg.content.clone();
                    messages.push(msg);
                    self.emit(AgentEvent::Done);
                    return Ok(content);
                }
                AgentResponse::ToolCalls(calls) => {
                    let assistant_msg = Message::assistant_tool_calls(calls.clone());
                    messages.push(assistant_msg);

                    for call in &calls {
                        self.emit(AgentEvent::ToolCallStart(call.clone()));

                        // ── before_tool_call hook ──
                        if let Some(ref hooks) = self.hooks {
                            match hooks.before_tool_call(&ctx, call).await {
                                HookResult::Abort(reason) => {
                                    self.emit(AgentEvent::Error(reason.clone()));
                                    return Err(AgentError::Other(reason));
                                }
                                HookResult::Skip => {
                                    messages.push(Message::tool_result(&call.id, "[skipped by hook]"));
                                    continue;
                                }
                                HookResult::Continue => {}
                            }
                        }

                        let result = match &self.tools {
                            Some(registry) => {
                                registry
                                    .call(&call.name, call.arguments.clone(), &self.process_manager)
                                    .await
                                    .unwrap_or_else(|e| format!("Tool error: {e}"))
                            }
                            None => format!("Tool '{}' not available", call.name),
                        };

                        // ── after_tool_call hook ──
                        let mut result = result;
                        if let Some(ref hooks) = self.hooks {
                            hooks.after_tool_call(&ctx, call, &mut result).await;
                        }

                        self.emit(AgentEvent::ToolCallResult {
                            call_id: call.id.clone(),
                            result: result.clone(),
                        });

                        messages.push(Message::tool_result(&call.id, &result));
                    }
                }
            }
        }

        self.emit(AgentEvent::Done);
        Ok(String::new())
    }

    #[inline]
    fn emit(&self, event: AgentEvent) {
        self.listener.on_event(&event);
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .as_ref()
            .map(|t| t.definitions())
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl AgentLike for BaseAgent {
    async fn run_turn(
        &self,
        provider: &ModelProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        listener: &dyn EventListener,
    ) -> Result<AgentResponse, AgentError> {
        self.adapter
            .chat(provider, messages, tools, listener)
            .await
    }

    fn max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
    }

    fn min_tier(&self) -> u8 {
        self.min_tier
    }

    fn adapter(&self) -> Arc<dyn LlmAdapter> {
        self.adapter.clone()
    }
}
