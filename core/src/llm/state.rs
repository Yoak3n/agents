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

#[derive(Debug, Clone)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}
