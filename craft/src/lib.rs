pub mod context;
pub mod cost;
pub mod memory;
pub mod prelude;
pub mod skill;
pub mod tools;

use std::pin::Pin;
use std::sync::Arc;

use yoakore::prelude::*;
use yoakore::provider::ProviderBalancer;

pub use context::{ContextConfig, ContextHook, ContextManager, DefaultContext};
pub use cost::{CostCalculator, CostTracker, PricingRule, PricingTable, ProviderUsage};
#[cfg(feature = "storage")]
pub use memory::MemoryStore;
pub use memory::{MemoryConfig, MemoryEntry, MemoryHook, MemoryProvider, NewMemory};
pub use skill::SkillManager;

type BuildResult = (BaseAgent, Option<Arc<CostTracker>>);

/// Full-featured agent built on yoakore.
///
/// Wraps `BaseAgent` with composable components (memory, context, cost)
/// and additional hooks. Use `CraftBuilder` to construct.
pub struct CraftAgent {
    inner: BaseAgent,
    provider: ModelProvider,
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

    /// Run with a [`ProviderBalancer`] for automatic provider selection.
    pub async fn run_balanced(
        &self,
        balancer: &ProviderBalancer,
        messages: &mut Vec<Message>,
    ) -> Result<String, AgentError> {
        self.inner.execute_balanced(balancer, messages).await
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
    providers: Vec<ModelProvider>,
    max_rounds: usize,
    tools: Option<ToolRegistry>,
    listener: Option<Arc<dyn EventListener>>,
    adapter: Option<Arc<dyn LlmAdapter>>,
    inject_rx: Option<tokio::sync::mpsc::Receiver<Message>>,

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
    approval: Option<ApprovalPolicy>,
}

impl CraftBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            provider: None,
            providers: Vec::new(),
            max_rounds: 200,
            tools: None,
            listener: None,
            adapter: None,
            inject_rx: None,
            memory: None,
            context: None,
            cost: None,
            memory_config: None,
            skill_manager: None,
            cost_tracker: None,
            extra_hooks: Vec::new(),
            file_tools: false,
            approval: None,
        }
    }

    // ── Core configuration ──

    /// Set the LLM provider (required).
    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set multiple providers for load balancing.
    ///
    /// When combined with `build_balanced()`, returns a [`ProviderBalancer`] for automatic selection.
    pub fn providers(mut self, providers: Vec<ModelProvider>) -> Self {
        self.providers = providers;
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

    /// Set an event listener from an Arc.
    pub fn listener_arc(mut self, listener: Arc<dyn EventListener>) -> Self {
        self.listener = Some(listener);
        self
    }

    /// Convenience: accept a closure for event callbacks.
    ///
    /// ```ignore
    /// CraftBuilder::new()
    ///     .provider(p)
    ///     .with_on_event(|event| {
    ///         if let AgentEvent::Delta(text) = event {
    ///             print!("{}", text);
    ///         }
    ///     })
    ///     .build();
    /// ```
    pub fn with_on_event(self, f: impl FnMut(&AgentEvent) + Send + Sync + 'static) -> Self {
        use std::sync::Mutex;

        struct FnEventListener<F: FnMut(&AgentEvent) + Send + Sync>(Mutex<F>);
        impl<F: FnMut(&AgentEvent) + Send + Sync> EventListener for FnEventListener<F> {
            fn on_event(&self, event: &AgentEvent) {
                if let Ok(mut f) = self.0.lock() {
                    f(event);
                }
            }
        }

        self.listener_arc(Arc::new(FnEventListener(Mutex::new(f))))
    }

    /// Set a channel receiver for injecting messages mid-run.
    ///
    /// Create the channel with `BaseAgent::inject_channel()`.
    pub fn inject_channel(mut self, rx: tokio::sync::mpsc::Receiver<Message>) -> Self {
        self.inject_rx = Some(rx);
        self
    }

    /// Register built-in file-system tools (`read_file`, `write_file`, `list_directory`, `search_files`).
    pub fn file_tools(mut self, enabled: bool) -> Self {
        self.file_tools = enabled;
        self
    }

    /// Register a single synchronous tool.
    ///
    /// Creates the [`ToolRegistry`] if it doesn't exist yet.
    pub fn tool(
        mut self,
        definition: ToolDefinition,
        handler: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        self.tools
            .get_or_insert_with(ToolRegistry::default)
            .register(definition, handler);
        self
    }

    /// Register a single async tool (with access to [`ProcessManager`]).
    ///
    /// Creates the [`ToolRegistry`] if it doesn't exist yet.
    pub fn tool_async(
        mut self,
        definition: ToolDefinition,
        handler: impl Fn(
            serde_json::Value,
            Arc<ProcessManager>,
        )
            -> Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.tools
            .get_or_insert_with(ToolRegistry::default)
            .register_async(definition, handler);
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

    /// Set a custom [`LlmAdapter`] (default is `OpenAIAdapter`).
    ///
    /// Use this to support non-OpenAI API formats.
    pub fn adapter(mut self, adapter: impl LlmAdapter + 'static) -> Self {
        self.adapter = Some(Arc::new(adapter));
        self
    }

    /// Set a custom [`LlmAdapter`] from an Arc.
    pub fn adapter_arc(mut self, adapter: Arc<dyn LlmAdapter>) -> Self {
        self.adapter = Some(adapter);
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

    /// Enable the memory system using [`MemoryStore`] (requires `storage` feature).
    #[cfg(feature = "storage")]
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
    pub fn hooks(mut self, hook: impl AgentHook + 'static) -> Self {
        self.extra_hooks.push(Arc::new(hook));
        self
    }

    /// Append a custom hook (Arc version) to the end of the hook chain.
    pub fn hooks_arc(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.extra_hooks.push(hook);
        self
    }

    /// Alias for [`hooks()`](Self::hooks).
    pub fn hook(self, hook: impl AgentHook + 'static) -> Self {
        self.hooks(hook)
    }

    /// Alias for [`hooks_arc()`](Self::hooks_arc).
    pub fn hook_arc(self, hook: Arc<dyn AgentHook>) -> Self {
        self.hooks_arc(hook)
    }

    /// Set an [`ApprovalPolicy`] to gate dangerous tool calls.
    ///
    /// The policy is applied to the [`ToolRegistry`] and checked by the agent
    /// before each tool execution.
    pub fn approval(mut self, policy: ApprovalPolicy) -> Self {
        self.approval = Some(policy);
        self
    }

    // ── Build ──

    /// Build the [`CraftAgent`] (single-provider).
    pub fn build(self) -> Result<CraftAgent, AgentError> {
        let provider = self
            .provider
            .clone()
            .ok_or_else(|| AgentError::Other("at least one provider is required".into()))?;

        let (inner, cost_tracker) = self.build_inner(vec![provider.clone()])?;

        Ok(CraftAgent {
            inner,
            provider,
            cost_tracker,
        })
    }

    /// Build with multiple providers, returning a [`ProviderBalancer`] for automatic selection.
    ///
    /// Use [`CraftAgent::run_balanced()`] with the returned balancer.
    pub fn build_balanced(self) -> Result<(CraftAgent, ProviderBalancer), AgentError> {
        let providers = if self.providers.is_empty() {
            vec![
                self.provider
                    .clone()
                    .ok_or_else(|| AgentError::Other("at least one provider is required".into()))?,
            ]
        } else {
            self.providers.clone()
        };

        let balancer = ProviderBalancer::new(providers.clone());
        let default_provider = providers[0].clone();
        let (inner, cost_tracker) = self.build_inner(providers)?;

        Ok((
            CraftAgent {
                inner,
                provider: default_provider,
                cost_tracker,
            },
            balancer,
        ))
    }

    /// Shared build logic.
    fn build_inner(self, providers: Vec<ModelProvider>) -> Result<BuildResult, AgentError> {
        let mut tools = self.tools.unwrap_or_default();

        if self.file_tools {
            tools::register_file_tools(&mut tools);
        }

        if let Some(policy) = self.approval {
            tools.set_approval(policy);
        }

        // Resolve memory: trait object takes precedence, otherwise create MemoryStore from config
        let memory_max_injected = self.memory_config.as_ref().map_or(5, |c| c.max_injected);
        #[cfg(feature = "storage")]
        let memory_provider: Option<Arc<dyn MemoryProvider>> = if let Some(provider) = self.memory {
            Some(provider)
        } else if let Some(config) = self.memory_config {
            Some(Arc::new(
                MemoryStore::new_default()
                    .expect("failed to create default memory store")
                    .auto_extract(config.auto_extract),
            ))
        } else {
            None
        };
        #[cfg(not(feature = "storage"))]
        let memory_provider: Option<Arc<dyn MemoryProvider>> = self.memory;

        // Use first provider's max_context_tokens for context hook
        let max_context_tokens = providers.first().map_or(128_000, |p| p.max_context_tokens);

        // Assemble hook chain: cost → context → memory → skills → extra_hooks
        let mut hooks: Vec<Arc<dyn AgentHook>> = Vec::new();

        if let Some(cost_calc) = self.cost {
            hooks.push(Arc::new(CostCalculatorHook(cost_calc)));
        }

        if let Some(ctx_mgr) = self.context {
            hooks.push(Arc::new(ContextHook::new(ctx_mgr, max_context_tokens)));
        }

        if let Some(mem_provider) = memory_provider {
            hooks.push(Arc::new(MemoryHook::new(mem_provider, memory_max_injected)));
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

        if let Some(adapter) = self.adapter {
            builder = builder.adapter_arc(adapter);
        }
        if let Some(hook) = composed_hook {
            builder = builder.hooks_arc(hook);
        }
        if let Some(listener) = self.listener {
            builder = builder.listener_arc(listener);
        }
        if let Some(rx) = self.inject_rx {
            builder = builder.inject_channel(rx);
        }

        builder = builder.providers(providers);
        let inner = builder.build_base();

        Ok((inner, self.cost_tracker))
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

        if let Some(skill_text) = self.0.format_for_prompt(query)
            && let Some(pos) = messages.iter().position(|m| m.role == Role::User)
        {
            messages.insert(pos, Message::system(skill_text));
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
