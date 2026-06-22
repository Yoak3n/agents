// ── Modules ──
pub mod agent;
pub mod hook;
pub mod provider;
pub mod runtime;
pub mod schema;
pub mod tools;
pub mod prelude;

mod error;
mod llm;
mod utils;

// ── Error ──
pub use error::AgentError;

// ── Schema: message, tool, config, event ──
pub use schema::common::{
    Message, Role,
    ToolCall, ToolDefinition,
    AppConfig, ApiStyle, ConfigError, EffortLevel, McpServerConfig, ModelKind, ModelProvider, ProviderGroup, ThinkingConfig, WorkspaceConfig,
    AgentEvent, EventListener, NullListener, ProcessStatus
};

// ── LLM ──
pub use llm::adapter::{AgentResponse, LlmAdapter};
pub use llm::state::{AgentState, Conversation};

// ── Agent ──
pub use agent::{AgentLike, ToolExecutor};
pub use agent::base::BaseAgent;
pub use agent::team::{
    TeamAgent, TeamAgentBuilder, TeamResult,
    CollaborativeAgent, CollaborativeAgentBuilder,
    AgentStep, AgentContact,
    AgentMessage, MessageType, ContactBook, MessageBus,
};
pub use agent::subagent::{SubAgent, SubAgentContext, SubAgentResult, SubAgentRegistry, SubAgentError};

// ── Hooks ──
pub use hook::{AgentHook, HookContext, HookResult};

// ── Provider ──
pub use provider::{
    EmbeddingAdapter, EmbeddingError, OllamaEmbeddingAdapter, OpenAIEmbeddingAdapter,
    OpenAIAdapter, ProviderBalancer, Semaphore, SemaphoreError,
    create_embedding_adapter,
};

// ── Tools ──
pub use tools::{ToolRegistry, ProcessManager};

// ── Utilities ──
pub use utils::{chunk_text, estimate_tokens};

// ── Extension ──
#[cfg(feature = "extension")]
pub use provider::embed_missing_documents;
#[cfg(feature = "extension")]
pub use runtime::selector::{HeuristicSelector, SkillSelector};
#[cfg(feature = "extension")]
pub use schema::extension::{
    Skill, SkillError, Storage, StorageError,
    SessionSummary, ConversationInfo, Summary, Document, DocumentSearchResult, MemoryEntry,
};
