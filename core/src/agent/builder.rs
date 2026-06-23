use std::sync::{Arc, Mutex};

use crate::agent::base::BaseAgent;
use crate::agent::plan::{PlanAgent, PlanResult};
use crate::agent::team::{CollaborativeAgent, TeamAgent, TeamAgentBuilder, TeamResult};
use crate::hook::AgentHook;
use crate::llm::LlmAdapter;
use crate::provider::{ProviderBalancer, RateLimitedAdapter, Semaphore};
use crate::schema::common::{AgentEvent, EventListener, Message, ModelProvider};
use crate::tools::ToolRegistry;

/// Wrapper that adapts a `FnMut(&AgentEvent)` closure into an `EventListener`.
struct FnEventListener<F: FnMut(&AgentEvent) + Send + Sync>(Mutex<F>);

impl<F: FnMut(&AgentEvent) + Send + Sync> EventListener for FnEventListener<F> {
    fn on_event(&self, event: &AgentEvent) {
        if let Ok(mut f) = self.0.lock() {
            f(event);
        }
    }
}

/// Unified output from any agent type.
pub enum AgentOutput {
    /// Simple text response (BaseAgent).
    Text(String),
    /// Team result with per-agent outputs and synthesis (TeamAgent).
    Team(TeamResult),
    /// Plan result with subtask outputs and synthesis (PlanAgent).
    Plan(PlanResult),
}

/// Unified builder for constructing different agent types.
///
/// All adapters are automatically wrapped with `RateLimitedAdapter` to enforce
/// per-provider rate limiting. The semaphore is configured from `ModelProvider.requests_per_minute`.
///
/// Uses `OpenAIAdapter` by default. Override with `.adapter()` for custom implementations.
///
/// ## Single provider
///
/// ```ignore
/// let agent = AgentBuilder::new()
///     .provider(provider)
///     .build_base();
/// ```
///
/// ## Multiple providers with load balancing
///
/// ```ignore
/// let (agent, balancer) = AgentBuilder::new()
///     .providers(vec![provider_a, provider_b, provider_c])
///     .build_base_balanced();
///
/// // Each call auto-selects a provider and rate-limits per provider
/// let reply = agent.execute_balanced(&balancer, &mut messages).await?;
/// ```
pub struct AgentBuilder {
    adapter: Option<Arc<dyn LlmAdapter>>,
    providers: Vec<ModelProvider>,
    semaphore: Arc<Semaphore>,
    max_rounds: usize,
    tools: Option<Arc<ToolRegistry>>,
    hooks: Option<Arc<dyn AgentHook>>,
    listener: Option<Arc<dyn EventListener>>,
    inject_rx: Option<Mutex<tokio::sync::mpsc::Receiver<Message>>>,
}

impl AgentBuilder {
    /// Create a new builder. Uses `OpenAIAdapter` by default.
    pub fn new() -> Self {
        Self {
            adapter: None,
            providers: Vec::new(),
            semaphore: Arc::new(Semaphore::new()),
            max_rounds: 200,
            tools: None,
            hooks: None,
            listener: None,
            inject_rx: None,
        }
    }

    /// Set a custom LLM adapter (default is `OpenAIAdapter`).
    pub fn adapter(mut self, adapter: impl LlmAdapter + 'static) -> Self {
        self.adapter = Some(Arc::new(adapter));
        self
    }

    /// Set a custom LLM adapter from an Arc.
    pub fn adapter_arc(mut self, adapter: Arc<dyn LlmAdapter>) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Set a single provider. Configures rate limiting from its `requests_per_minute`.
    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.semaphore.configure(&provider);
        self.providers = vec![provider];
        self
    }

    /// Set multiple providers. All are configured in the rate limiter.
    ///
    /// Use with `build_base_balanced()` to get a `ProviderBalancer` for automatic selection.
    pub fn providers(mut self, providers: Vec<ModelProvider>) -> Self {
        for p in &providers {
            self.semaphore.configure(p);
        }
        self.providers = providers;
        self
    }

    pub fn max_rounds(mut self, max: usize) -> Self {
        self.max_rounds = max;
        self
    }

    pub fn tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn hooks(mut self, hooks: impl AgentHook + 'static) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    pub fn hooks_arc(mut self, hooks: Arc<dyn AgentHook>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    pub fn listener(mut self, listener: impl EventListener + 'static) -> Self {
        self.listener = Some(Arc::new(listener));
        self
    }

    pub fn listener_arc(mut self, listener: Arc<dyn EventListener>) -> Self {
        self.listener = Some(listener);
        self
    }

    /// Convenience: accept a closure for event callbacks.
    ///
    /// ```ignore
    /// AgentBuilder::new()
    ///     .with_on_event(|event| {
    ///         if let AgentEvent::Delta(text) = event {
    ///             print!("{}", text);
    ///         }
    ///     })
    ///     .provider(provider)
    ///     .build_base()
    /// ```
    pub fn with_on_event(mut self, f: impl FnMut(&AgentEvent) + Send + Sync + 'static) -> Self {
        self.listener = Some(Arc::new(FnEventListener(Mutex::new(f))));
        self
    }

    /// Set a channel receiver for injecting messages mid-run.
    ///
    /// Create the channel with `BaseAgent::inject_channel()`.
    pub fn inject_channel(mut self, rx: tokio::sync::mpsc::Receiver<Message>) -> Self {
        self.inject_rx = Some(Mutex::new(rx));
        self
    }

    /// Resolve the adapter, always wrapped with rate limiting.
    fn resolve_adapter(&self) -> Arc<dyn LlmAdapter> {
        let inner = self
            .adapter
            .clone()
            .unwrap_or_else(|| Arc::new(crate::provider::OpenAIAdapter::new()));
        Arc::new(RateLimitedAdapter::new(inner, self.semaphore.clone()))
    }

    fn apply_options(self, mut agent: BaseAgent) -> BaseAgent {
        if let Some(tools) = self.tools {
            agent = agent.with_tools(tools);
        }
        if let Some(hooks) = self.hooks {
            agent = agent.with_hooks_arc(hooks);
        }
        if let Some(listener) = self.listener {
            agent = agent.with_listener_arc(listener);
        }
        if let Some(rx) = self.inject_rx {
            agent = agent.with_inject_channel(rx.into_inner().unwrap());
        }
        agent
    }

    /// Build a BaseAgent with the configured common properties.
    ///
    /// For single-provider use. For multiple providers, use `build_base_balanced()`.
    pub fn build_base(self) -> BaseAgent {
        let adapter = self.resolve_adapter();
        let agent = BaseAgent::new_arc(adapter).with_max_rounds(self.max_rounds);
        self.apply_options(agent)
    }

    /// Build a BaseAgent and a ProviderBalancer for multi-provider setups.
    ///
    /// The balancer selects providers automatically. Rate limiting is enforced
    /// per-provider via the shared Semaphore.
    pub fn build_base_balanced(self) -> (BaseAgent, ProviderBalancer) {
        let balancer = ProviderBalancer::new(self.providers.clone());
        let adapter = self.resolve_adapter();
        let agent = BaseAgent::new_arc(adapter).with_max_rounds(self.max_rounds);
        (self.apply_options(agent), balancer)
    }

    /// Build a TeamAgent from the given CollaborativeAgents.
    ///
    /// If multiple providers were set via `providers()`, the team uses
    /// `build_balanced()` internally for automatic load balancing.
    pub fn build_team(self, agents: Vec<CollaborativeAgent>) -> TeamAgent {
        let adapter = self.resolve_adapter();
        let mut builder = TeamAgentBuilder::default().max_rounds(self.max_rounds);

        for agent in agents {
            builder = builder.add_agent(agent);
        }

        if self.providers.len() > 1 {
            builder = builder.providers(self.providers);
            builder.build_balanced(adapter)
        } else {
            if let Some(provider) = self.providers.into_iter().next() {
                builder = builder.provider(provider);
            }
            builder.build(adapter)
        }
    }

    /// Build a PlanAgent with the configured common properties.
    ///
    /// If multiple providers were set, the plan agent uses load balancing.
    pub fn build_plan(self) -> PlanAgent {
        let adapter = self.resolve_adapter();
        PlanAgent::new(adapter)
            .max_plan_rounds(self.max_rounds)
            .providers(self.providers)
    }

    /// Build a CollaborativeAgent suitable for use with `build_team()`.
    ///
    /// This is a convenience helper so callers don't need to import
    /// `CollaborativeAgentBuilder` separately.
    pub fn team_agent(
        adapter: &Arc<dyn LlmAdapter>,
        name: impl Into<String>,
        role: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> CollaborativeAgent {
        use crate::agent::team::CollaborativeAgentBuilder;

        CollaborativeAgentBuilder::default()
            .name(name)
            .role(role)
            .system_prompt(system_prompt)
            .build(adapter.clone())
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
