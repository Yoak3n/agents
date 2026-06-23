use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::schema::common::Message;

#[derive(Debug, Clone, Default)]
pub enum AgentState {
    #[default]
    Idle,
    Thinking,
    Streaming {
        partial: String,
    },
    UsingTool {
        tool_name: String,
    },
    Error {
        message: String,
    },
}

/// Manages a conversation history with optional system prompt.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub messages: Vec<Message>,
    system_prompt: Option<String>,
}

impl Conversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
        }
    }

    /// Create a conversation with a system prompt.
    pub fn with_system(prompt: impl Into<String>) -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: Some(prompt.into()),
        }
    }

    /// Set or replace the system prompt.
    pub fn set_system(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    /// Get the system prompt, if set.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Get all messages including the system prompt as the first message.
    ///
    /// If a system prompt is set, it is prepended to the message list.
    /// The returned vec is suitable for passing directly to the LLM.
    pub fn messages_with_system(&self) -> Vec<Message> {
        match &self.system_prompt {
            Some(prompt) => {
                let mut msgs = Vec::with_capacity(self.messages.len() + 1);
                msgs.push(Message::system(prompt));
                msgs.extend(self.messages.iter().cloned());
                msgs
            }
            None => self.messages.clone(),
        }
    }

    /// Push a message.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Add a user message.
    pub fn add_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    /// Add an assistant message.
    pub fn add_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }

    /// Get the last message, if any.
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Number of messages (excluding system prompt).
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Keep only the last N messages. Useful for context window management.
    pub fn truncate(&mut self, max_messages: usize) {
        if self.messages.len() > max_messages {
            let drain_count = self.messages.len() - max_messages;
            self.messages.drain(..drain_count);
        }
    }

    /// Clear all messages (but keep the system prompt).
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

/// A named session wrapping a Conversation with metadata.
///
/// This is the recommended entry point for multi-turn conversations.
///
/// ```rust
/// use yoakore::Session;
///
/// let mut session = Session::with_system("You are a helpful assistant.");
/// session.add_user("Hello!");
/// // session.messages_with_system() returns [system, user] for the LLM
/// ```
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub conversation: Conversation,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            conversation: Conversation::new(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a session with a system prompt.
    pub fn with_system(prompt: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            conversation: Conversation::with_system(prompt),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set or replace the system prompt.
    pub fn set_system(&mut self, prompt: impl Into<String>) {
        self.conversation.set_system(prompt);
    }

    /// Add a user message.
    pub fn add_user(&mut self, content: impl Into<String>) {
        self.conversation.add_user(content);
    }

    /// Add an assistant message.
    pub fn add_assistant(&mut self, content: impl Into<String>) {
        self.conversation.add_assistant(content);
    }

    /// Get all messages including system prompt.
    pub fn messages_with_system(&self) -> Vec<Message> {
        self.conversation.messages_with_system()
    }

    /// Number of messages (excluding system prompt).
    pub fn len(&self) -> usize {
        self.conversation.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conversation.is_empty()
    }

    /// Keep only the last N messages.
    pub fn truncate(&mut self, max_messages: usize) {
        self.conversation.truncate(max_messages);
    }

    /// Clear all messages (keep system prompt and metadata).
    pub fn clear(&mut self) {
        self.conversation.clear();
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
