//! Convenience re-exports for common usage.
//!
//! ```rust
//! use agent_core::prelude::*;
//! ```

// Fundamental message types
pub use crate::schema::common::{Message, Role, ToolCall, ToolDefinition};

// Config
pub use crate::schema::common::{AppConfig, ApiStyle, ConfigError, EffortLevel, ModelKind, ModelProvider, ProviderGroup, ThinkingConfig};

// Events
pub use crate::schema::common::{AgentEvent, EventListener, NullListener, ProcessStatus};

// LLM
pub use crate::llm::adapter::{AgentResponse, LlmAdapter};

// Agent
pub use crate::agent::{AgentLike, ToolExecutor};
pub use crate::agent::base::BaseAgent;

// Hooks
pub use crate::hook::{AgentHook, HookContext, HookResult};

// Error
pub use crate::error::AgentError;

// Tools
pub use crate::tools::{ToolRegistry, ProcessManager};

// State
pub use crate::llm::state::{AgentState, Conversation};

// Provider
pub use crate::provider::OpenAIAdapter;

// Extension types (feature-gated)
#[cfg(feature = "extension")]
pub use crate::schema::extension::{
    Skill, SkillError, Storage, StorageError,
    SessionSummary, MemoryEntry,
};
