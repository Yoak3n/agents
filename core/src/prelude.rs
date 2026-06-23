//! Convenience re-exports for common usage.
//!
//! ```rust
//! use yoakore::prelude::*;
//! ```

// Fundamental message types
pub use crate::schema::common::{Message, Role, ToolCall, ToolDefinition};

// Config
pub use crate::schema::common::{
    ApiStyle, AppConfig, ConfigError, EffortLevel, ModelKind, ModelProvider, ProviderGroup,
    ThinkingConfig,
};

// Events
pub use crate::schema::common::{AgentEvent, EventListener, NullListener, ProcessStatus};

// LLM
pub use crate::llm::adapter::{AgentResponse, AgentResponseKind, LlmAdapter, Usage};

// Agent
pub use crate::agent::base::BaseAgent;
pub use crate::agent::builder::{AgentBuilder, AgentOutput};
pub use crate::agent::plan::{PlanAgent, PlanResult, Subtask};
pub use crate::agent::subagent::{SubAgent, SubAgentContext, SubAgentRegistry, SubAgentResult};
pub use crate::agent::team::{
    CollaborativeAgent, CollaborativeAgentBuilder, TeamAgent, TeamResult,
};
pub use crate::agent::{AgentLike, ToolExecutor};

// Hooks
pub use crate::hook::{AgentHook, ComposedHook, HookContext, HookResult};

// Tool approval
pub use crate::tools::{ApprovalCallback, ApprovalPolicy, ApprovalPolicyBuilder};

// Error
pub use crate::error::AgentError;

// Tools
pub use crate::tools::{ProcessManager, ToolRegistry};

// State
pub use crate::llm::state::{AgentState, Conversation, Session};

// Provider
pub use crate::provider::OpenAIAdapter;

// Extension: skills (feature-gated)
#[cfg(feature = "skills")]
pub use crate::schema::extension::{Skill, SkillError};

// Extension: storage (feature-gated)
#[cfg(feature = "storage")]
pub use crate::schema::extension::{SessionSummary, Storage, StorageError};
