// ── Modules ──
pub mod agent;
pub mod hook;
pub mod prelude;
pub mod provider;
pub mod runtime;
pub mod schema;
pub mod tools;

mod error;
mod llm;
mod utils;

// ── Error ──
pub use error::AgentError;

// ── Schema: message, tool, config, event ──
pub use schema::common::{
    AgentEvent, ApiStyle, AppConfig, ConfigError, EffortLevel, EventListener, McpServerConfig,
    Message, ModelKind, ModelProvider, NullListener, ProcessStatus, ProviderGroup, Role,
    ThinkingConfig, ToolCall, ToolDefinition, WorkspaceConfig,
};

// ── LLM ──
pub use llm::adapter::{AgentResponse, AgentResponseKind, LlmAdapter, Usage};
pub use llm::state::{AgentState, Conversation, Session};

// ── Agent ──
pub use agent::base::BaseAgent;
pub use agent::builder::{AgentBuilder, AgentOutput};
pub use agent::plan::{PlanAgent, PlanAgentBuilder, PlanResult, Subtask};
pub use agent::subagent::{
    SubAgent, SubAgentContext, SubAgentError, SubAgentRegistry, SubAgentResult, SubAgentStatus,
};
pub use agent::team::{
    AgentContact, AgentStep, CollaborativeAgent, CollaborativeAgentBuilder, ContactBook, TeamAgent,
    TeamAgentBuilder, TeamResult,
};
pub use agent::{AgentLike, ToolExecutor};

// ── Hooks ──
pub use hook::{AgentHook, ComposedHook, HookContext, HookResult};

// ── Provider ──
pub use provider::{
    EmbeddingAdapter, EmbeddingError, OllamaEmbeddingAdapter, OpenAIAdapter,
    OpenAIEmbeddingAdapter, ProviderBalancer, RateLimitedAdapter, Semaphore, SemaphoreError,
    create_embedding_adapter,
};

// ── Tools ──
pub use tools::{
    ApprovalCallback, ApprovalPolicy, ApprovalPolicyBuilder, ProcessManager, ToolRegistry,
};

// ── Utilities ──
pub use utils::{chunk_text, estimate_tokens};

// ── Extension: skills ──
#[cfg(feature = "skills")]
pub use runtime::selector::{HeuristicSelector, SkillSelector};
#[cfg(feature = "skills")]
pub use schema::extension::{Skill, SkillError};

// ── Extension: storage ──
#[cfg(feature = "storage")]
pub use provider::embed_missing_documents;
#[cfg(feature = "storage")]
pub use schema::extension::{
    ConversationInfo, Document, DocumentSearchResult, SessionSummary, Storage, StorageError,
    Summary,
};
