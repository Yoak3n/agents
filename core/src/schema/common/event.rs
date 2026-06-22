use serde::{Deserialize, Serialize};

use crate::schema::common::{Message, tool::ToolCall};

/// Event listener — receives `AgentEvent` callbacks during agent execution.
///
/// Implement this trait to handle streaming output, tool call notifications, etc.
/// The adapter emits `Delta` events for streaming text; the orchestrator emits
/// `Thinking`, `ToolCallStart`, `ToolCallResult`, `Done`, `Error`, etc.
pub trait EventListener: Send + Sync {
    fn on_event(&self, event: &AgentEvent);
}

/// A no-op listener that ignores all events.
pub struct NullListener;
impl EventListener for NullListener {
    fn on_event(&self, _event: &AgentEvent) {}
}

#[cfg(feature = "extension")]
use crate::schema::extension::storage::SessionSummary;
/// Status of a managed subprocess
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessStatus {
    Running,
    Exited(i32),
    Killed,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// Agent started thinking
    Thinking,
    /// Agent produced a partial response (streaming)
    Delta(String),
    /// Reasoning/thinking content delta (streamed from LLM)
    ThinkingDelta(String),
    /// Agent completed a full message
    MessageComplete(Message),
    /// Agent wants to call a tool
    ToolCallStart(ToolCall),
    /// Tool returned a result
    ToolCallResult { call_id: String, result: String },
    /// Streaming output from a managed subprocess (stdout or stderr)
    ProcessOutput { call_id: String, line: String },
    /// Status change of a managed subprocess
    ProcessStatus {
        call_id: String,
        status: ProcessStatus,
    },
    /// Agent encountered an error
    Error(String),
    /// Agent finished processing
    Done,
    // ── Session management events ──
    /// Session list loaded

    #[cfg(feature = "extension")]
    SessionsLoaded(Vec<SessionSummary>),
    /// New session created (session_id)
    SessionCreated(String),
    /// Switched to a session (session_id)
    SessionSwitched(String),
    /// Messages loaded for a session
    MessagesLoaded {
        session_id: String,
        messages: Vec<Message>,
    },
    /// More messages loaded for infinite scroll
    MoreMessagesLoaded {
        session_id: String,
        messages: Vec<Message>,
        has_more: bool,
    },
}
