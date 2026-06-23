pub mod context;
pub mod cost;
pub mod memory;
pub mod prelude;
pub mod skill;
pub mod tools;

use std::sync::Arc;

use yoakore::Storage;
use yoakore::prelude::*;

pub use context::{ContextConfig, ContextHook, ContextManager, DefaultContext};
pub use cost::{CostCalculator, CostTracker, PricingRule, PricingTable, ProviderUsage};
pub use memory::{DefaultMemory, MemoryConfig, MemoryHook, MemoryProvider, NewMemory};
pub use skill::SkillManager;

/// Full-featured agent built on yoakore.
///
/// Wraps `BaseAgent` with composable components (memory, context, cost)
/// and additional hooks. Use `CraftBuilder` to construct.
pub struct CraftAgent {
    inner: BaseAgent,
    provider: ModelProvider,
    storage: Arc<Storage>,
    cost_tracker: Option<Arc<CostTracker>>,
}

impl CraftAgent {
    /// Run the agent loop with the given message list.
    ///
    /// Appends the user message, calls the LLM, executes any tool calls,
    /// and returns the final assistant reply.
    pub async fn run(&self, messages: &mut Vec<Message>) -> Result<String, AgentError> {
        self.inner.execute(&self.provider, messages).await
    }

    /// Run the agent loop within a [`Session`].
    ///
    /// Convenience wrapper that delegates to the session's internal message list.
    pub async fn run_in_session(&self, session: &mut Session) -> Result<String, AgentError> {
        self.inner.execute_in_session(&self.provider, session).await
    }

    /// Access the underlying [`Storage`] handle.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// Borrow the inner [`BaseAgent`].
    pub fn inner(&self) -> &BaseAgent {
        &self.inner
    }

    /// Mutably borrow the inner [`BaseAgent`].
    pub fn inner_mut(&mut self) -> &mut BaseAgent {
        &mut self.inner
    }

    /// Access the [`CostTracker`], if cost tracking was enabled.
    pub fn cost_tracker(&self) -> Option<&Arc<CostTracker>> {
        self.cost_tracker.as_ref()
    }
}

/// Fluent builder for `CraftAgent`.
///
/// Supports two customization patterns:
///
/// **Trait objects** — replace core components:
/// ```ignore
/// CraftBuilder::new()
///     .provider(p)
///     .memory_provider(Arc::new(MyMemory))
///     .context_manager(Arc::new(MyContext))
///     .build();
/// ```
///
/// **Convenience methods** — use default implementations:
/// ```ignore
/// CraftBuilder::new()
///     .provider(p)
///     .memory(MemoryConfig::default())
///     .context(ContextConfig::default())
///     .cost_tracking(PricingTable::new().default(PricingRule::new(0.001, 0.002)))
///     .build();
/// ```
pub struct CraftBuilder {
    provider: Option<ModelProvider>,
    storage: Option<Arc<Storage>>,
    max_rounds: usize,
    tools: Option<ToolRegistry>,
    listener: Option<Arc<dyn EventListener>>,

    // 可替换组件 (trait objects)
    memory: Option<Arc<dyn MemoryProvider>>,
    context: Option<Arc<dyn ContextManager>>,
    cost: Option<Arc<dyn CostCalculator>>,

    // 便捷方法的 config（与 trait object 互斥）
    memory_config: Option<MemoryConfig>,

    // Skills
    skill_manager: Option<SkillManager>,

    // CostTracker reference for CraftAgent to expose
    cost_tracker: Option<Arc<CostTracker>>,

    // 额外 hooks
    extra_hooks: Vec<Arc<dyn AgentHook>>,

    file_tools: bool,
}

impl CraftBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            provider: None,
            storage: None,
            max_rounds: 200,
            tools: None,
            listener: None,
            memory: None,
            context: None,
            cost: None,
            memory_config: None,
            skill_manager: None,
            cost_tracker: None,
            extra_hooks: Vec::new(),
            file_tools: false,
        }
    }

    // ── Core configuration ──

    /// Set the LLM provider (required).
    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set a custom [`Storage`] backend. Defaults to an in-process SQLite database.
    pub fn storage(mut self, storage: Arc<Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the maximum number of tool-call rounds per LLM turn. Default: 200.
    pub fn max_rounds(mut self, max: usize) -> Self {
        self.max_rounds = max;
        self
    }

    /// Set the tool registry.
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set an event listener for streaming output.
    pub fn listener(mut self, listener: impl EventListener + 'static) -> Self {
        self.listener = Some(Arc::new(listener));
        self
    }

    /// Register built-in file-system tools (`read_file`, `write_file`, `list_directory`, `search_files`).
    pub fn file_tools(mut self, enabled: bool) -> Self {
        self.file_tools = enabled;
        self
    }

    /// Load skills from a workspace directory.
    pub fn skills(mut self, workspace: impl Into<std::path::PathBuf>) -> Self {
        self.skill_manager = Some(SkillManager::new(workspace));
        self
    }

    /// Set an existing `SkillManager` instance.
    pub fn skill_manager(mut self, manager: SkillManager) -> Self {
        self.skill_manager = Some(manager);
        self
    }

    // ── Trait-object replacement ──

    /// Replace the default memory implementation with a custom [`MemoryProvider`].
    pub fn memory_provider(mut self, provider: Arc<dyn MemoryProvider>) -> Self {
        self.memory = Some(provider);
        self.memory_config = None;
        self
    }

    /// Replace the default context manager with a custom [`ContextManager`].
    pub fn context_manager(mut self, manager: Arc<dyn ContextManager>) -> Self {
        self.context = Some(manager);
        self
    }

    /// Replace the default cost calculator with a custom [`CostCalculator`].
    pub fn cost_calculator(mut self, calc: Arc<dyn CostCalculator>) -> Self {
        self.cost = Some(calc);
        self
    }

    // ── Convenience methods (use default implementations) ──

    /// Enable the memory system using [`DefaultMemory`].
    pub fn memory(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self.memory = None;
        self
    }

    /// Enable context management using [`DefaultContext`].
    pub fn context(mut self, config: ContextConfig) -> Self {
        self.context = Some(Arc::new(DefaultContext::new(config)));
        self
    }

    /// Enable cost tracking using [`CostTracker`].
    pub fn cost_tracking(mut self, pricing: PricingTable) -> Self {
        let tracker = Arc::new(CostTracker::new(pricing));
        self.cost_tracker = Some(tracker.clone());
        self.cost = Some(tracker);
        self
    }

    // ── Extra hooks ──

    /// Append a custom hook to the end of the hook chain.
    pub fn hook(mut self, hook: impl AgentHook + 'static) -> Self {
        self.extra_hooks.push(Arc::new(hook));
        self
    }

    /// Append a custom hook (Arc version) to the end of the hook chain.
    pub fn hook_arc(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.extra_hooks.push(hook);
        self
    }

    // ── Build ──

    pub fn build(self) -> Result<CraftAgent, AgentError> {
        let provider = self
            .provider
            .ok_or_else(|| AgentError::Other("at least one provider is required".into()))?;

        let storage = self
            .storage
            .unwrap_or_else(|| Arc::new(Storage::new().expect("failed to create default storage")));

        let mut tools = self.tools.unwrap_or_default();

        if self.file_tools {
            tools::register_file_tools(&mut tools);
        }

        // Resolve memory: trait object takes precedence, otherwise create DefaultMemory from config
        let memory_max_injected = self.memory_config.as_ref().map_or(5, |c| c.max_injected);
        let memory_provider: Option<Arc<dyn MemoryProvider>> = if let Some(provider) = self.memory {
            Some(provider)
        } else if let Some(config) = self.memory_config {
            Some(Arc::new(
                DefaultMemory::new(storage.clone()).auto_extract(config.auto_extract),
            ))
        } else {
            None
        };

        // Assemble hook chain: cost → context → memory → skills → extra_hooks
        let mut hooks: Vec<Arc<dyn AgentHook>> = Vec::new();

        if let Some(cost_calc) = self.cost {
            hooks.push(Arc::new(CostCalculatorHook(cost_calc)));
        }

        if let Some(ctx_mgr) = self.context {
            hooks.push(Arc::new(ContextHook::new(
                ctx_mgr,
                provider.max_context_tokens,
            )));
        }

        if let Some(mem_provider) = memory_provider {
            hooks.push(Arc::new(MemoryHook::new(
                mem_provider,
                storage.clone(),
                memory_max_injected,
            )));
        }

        if let Some(skill_mgr) = self.skill_manager {
            hooks.push(Arc::new(SkillHook(skill_mgr)));
        }

        hooks.extend(self.extra_hooks);

        let composed_hook = compose_hooks(hooks);

        // Build agent
        let mut builder = AgentBuilder::new()
            .max_rounds(self.max_rounds)
            .tools(Arc::new(tools));

        if let Some(hook) = composed_hook {
            builder = builder.hooks_arc(hook);
        }
        if let Some(listener) = self.listener {
            builder = builder.listener_arc(listener);
        }

        let inner = builder.provider(provider.clone()).build_base();

        Ok(CraftAgent {
            inner,
            provider,
            storage,
            cost_tracker: self.cost_tracker,
        })
    }
}

impl Default for CraftBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──

/// Wraps a [`CostCalculator`] as an [`AgentHook`].
struct CostCalculatorHook(Arc<dyn CostCalculator>);

#[async_trait::async_trait]
impl AgentHook for CostCalculatorHook {
    async fn after_llm_call(
        &self,
        ctx: &HookContext<'_>,
        response: &mut yoakore::AgentResponse,
    ) -> HookResult {
        use yoakore::AgentResponseKind;

        let provider_name = &ctx.provider.name;

        if let Some(ref usage) = response.usage {
            self.0
                .record_usage(
                    provider_name,
                    usage.prompt_tokens,
                    usage.cached_input_tokens,
                    usage.completion_tokens,
                )
                .await;
        } else {
            let output_tokens = match &response.kind {
                AgentResponseKind::MessageComplete(msg) => {
                    yoakore::estimate_tokens(&msg.content) as u64
                }
                AgentResponseKind::ToolCalls(_) => 0,
            };
            self.0
                .record_usage(provider_name, 0, 0, output_tokens)
                .await;
        }

        HookResult::Continue
    }
}

/// Wraps [`SkillManager`] as an [`AgentHook`] — injects matching skills before LLM calls.
struct SkillHook(SkillManager);

#[async_trait::async_trait]
impl AgentHook for SkillHook {
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

        if let Some(skill_text) = self.0.format_for_prompt(query) {
            if let Some(pos) = messages.iter().position(|m| m.role == Role::User) {
                messages.insert(pos, Message::system(skill_text));
            }
        }

        HookResult::Continue
    }
}

/// Compose a list of hooks into a single chained hook.
fn compose_hooks(hooks: Vec<Arc<dyn AgentHook>>) -> Option<Arc<dyn AgentHook>> {
    if hooks.is_empty() {
        return None;
    }
    let mut iter = hooks.into_iter();
    let first = iter.next().unwrap();
    Some(iter.fold(first, |acc, next| {
        Arc::new(ComposedHook {
            first: acc,
            second: next,
        })
    }))
}
