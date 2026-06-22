use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::mpsc;

use super::message::AgentMessage;

/// Message bus — routes messages between registered agents.
///
/// Uses interior mutability so it can be shared via `Arc<MessageBus>`.
/// Registration happens during setup; `send` is lock-free for runtime use.
pub struct MessageBus {
    mailboxes: Mutex<HashMap<String, mpsc::UnboundedSender<AgentMessage>>>,
}

impl MessageBus {
    pub fn new() -> Self {
        Self {
            mailboxes: Mutex::new(HashMap::new()),
        }
    }

    /// Register an agent's message sender under its name.
    /// Call this during setup before sharing the bus.
    pub fn register_agent(&self, name: String, sender: mpsc::UnboundedSender<AgentMessage>) {
        self.mailboxes.lock().unwrap().insert(name, sender);
    }

    /// Send a message to the agent named in `msg.to`.
    pub async fn send(&self, msg: AgentMessage) -> Result<(), String> {
        let sender = {
            let mailboxes = self.mailboxes.lock().unwrap();
            mailboxes.get(&msg.to).cloned()
        };
        if let Some(sender) = sender {
            sender.send(msg).map_err(|e| format!("Failed to send message: {e}"))
        } else {
            Err(format!("Agent '{}' not found", msg.to))
        }
    }

    /// List all registered agent names.
    pub fn agent_names(&self) -> Vec<String> {
        self.mailboxes.lock().unwrap().keys().cloned().collect()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}
