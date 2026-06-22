use serde::{Deserialize, Serialize};

/// Message sent between agents via the message bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Sender agent name
    pub from: String,
    /// Receiver agent name
    pub to: String,
    /// Message type for routing/filtering
    pub msg_type: MessageType,
    /// Task description or content
    pub content: String,
    /// Optional conversation ID to track related messages
    pub conversation_id: String,
}

/// Message types for task classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// Request for information or analysis
    Query,
    /// Request to perform a specific task
    Task,
    /// Response to a query or task
    Response,
    /// Request for collaboration/input from another agent
    Collaboration,
    /// Status update or notification
    Status,
}

/// Response from an agent back to the coordinator
#[derive(Debug)]
pub struct AgentResponseMessage {
    pub from: String,
    pub content: String,
    pub success: bool,
}
