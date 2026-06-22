use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use crate::schema::common::{Message, ModelProvider, NullListener, ToolCall, ToolDefinition};

use crate::llm::{AgentResponse, LlmAdapter};
use super::bus::MessageBus;

/// Contact book — maps agent names to their capabilities for routing decisions.
#[derive(Debug, Clone, Default)]
pub struct ContactBook {
    pub contacts: HashMap<String, AgentContact>,
}

impl ContactBook {
    pub fn new() -> Self {
        Self { contacts: HashMap::new() }
    }

    pub fn add(&mut self, name: String, role: String, capabilities: Vec<String>, description: String) {
        self.contacts.insert(name.clone(), AgentContact { name, role, capabilities, description });
    }

    /// Render all contacts as a human-readable string for system prompts.
    pub fn render(&self) -> String {
        self.contacts.values()
            .map(|c| format!("- {} ({}): {}", c.name, c.role, c.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone)]
pub struct AgentContact {
    pub name: String,
    pub role: String,
    pub capabilities: Vec<String>,
    pub description: String,
}

/// A collaborative agent that participates in team tasks.
///
/// Agents communicate exclusively through tools:
/// - `report_result` — the **only** way to return work to the coordinator
/// - `ask_peer` — request information from another team member
///
/// This enforces the constraint that agents can only contribute their own
/// domain work, and can never synthesize others' outputs.
pub struct CollaborativeAgent {
    name: String,
    role: String,
    system_prompt: String,
    adapter: Arc<dyn LlmAdapter>,
    contact_book: ContactBook,
    provider: Option<ModelProvider>,
    #[allow(dead_code)]
    bus: Arc<MessageBus>,
}

/// What the agent wants to do next in its processing loop.
pub enum AgentStep {
    /// Agent called `report_result` — work is done.
    Done(String),
    /// Agent called `ask_peer` — needs coordinator to route the question.
    AskPeer { peer: String, question: String },
    /// LLM returned text without tools — treat as final output.
    TextOutput(String),
    /// Error occurred.
    Error(String),
}

impl CollaborativeAgent {
    pub fn builder() -> CollaborativeAgentBuilder {
        CollaborativeAgentBuilder::default()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    /// Get the tool definitions this agent supports.
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let peer_names: Vec<String> = self.contact_book.contacts.keys().cloned().collect();

        vec![
            ToolDefinition {
                name: "report_result".to_string(),
                description: "Submit your final work output to the coordinator. \
                    This is the ONLY way to return your result. \
                    Call this when you have completed your analysis or task."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "Your complete work output — analysis, findings, or task result."
                        }
                    },
                    "required": ["result"]
                }),
            },
            ToolDefinition {
                name: "ask_peer".to_string(),
                description: format!(
                    "Request information from another team member. \
                    Available peers: {}. \
                    The coordinator routes your question and feeds the answer back to you.",
                    peer_names.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "peer": {
                            "type": "string",
                            "description": "Name of the team member to ask.",
                            "enum": peer_names
                        },
                        "question": {
                            "type": "string",
                            "description": "Specific question to ask."
                        }
                    },
                    "required": ["peer", "question"]
                }),
            },
        ]
    }

    /// Build the system prompt and initial messages for a task.
    pub fn build_task_messages(&self, task: &str) -> Vec<Message> {
        let contacts = self.contact_book.render();

        let system = format!(
            "{prompt}\n\n\
            You are **{name}**, a **{role}** specialist on a team.\n\n\
            ## Your job\n\
            Complete the task using your domain expertise. \
            When done, call `report_result` with your output.\n\n\
            ## Rules\n\
            - Only contribute work in YOUR domain ({role})\n\
            - Use `ask_peer` if you need information from other specialists\n\
            - Always call `report_result` when finished — this is the ONLY way to submit your work\n\
            - Do NOT attempt to synthesize or summarize other agents' work\n\n\
            ## Team members\n{contacts}",
            prompt = self.system_prompt,
            name = self.name,
            role = self.role,
            contacts = contacts,
        );

        vec![
            Message::system(&system),
            Message::user(task),
        ]
    }

    /// Run one LLM turn with tools, returning what the agent wants to do next.
    ///
    /// The coordinator drives the loop:
    /// 1. Call `run_step(messages)` → get `AgentStep`
    /// 2. If `Done` → collect result
    /// 3. If `AskPeer` → route question, inject answer into messages, call again
    /// 4. If `TextOutput` → treat as result
    pub async fn run_step(&self, messages: &mut Vec<Message>) -> AgentStep {
        let tools = self.tool_definitions();

        let provider = match self.select_provider() {
            Some(p) => p,
            None => return AgentStep::Error("no available provider".to_string()),
        };

        match self.adapter.chat(&provider, messages, &tools, &NullListener).await {
            Ok(AgentResponse::MessageComplete(msg)) => {
                AgentStep::TextOutput(msg.content)
            }
            Ok(AgentResponse::ToolCalls(calls)) => {
                // Check for report_result
                if let Some(result) = self.find_report_result(&calls) {
                    return AgentStep::Done(result);
                }

                // Check for ask_peer
                if let Some((peer, question)) = self.find_ask_peer(&calls) {
                    // Add assistant message to history
                    messages.push(Message::assistant_tool_calls(calls));
                    return AgentStep::AskPeer { peer, question };
                }

                // Other tool calls — add to messages and continue
                messages.push(Message::assistant_tool_calls(calls.clone()));
                for call in &calls {
                    let output = self.execute_tool(call).await;
                    messages.push(Message::tool_result(&call.id, &output));
                }
                // Return a signal to continue (coordinator will call again)
                AgentStep::TextOutput(String::new()) // empty = continue
            }
            Err(e) => AgentStep::Error(e.to_string()),
        }
    }

    /// Inject a peer's answer into the conversation and continue.
    ///
    /// Called by the coordinator after routing an `ask_peer` request.
    pub fn inject_peer_answer(&self, messages: &mut Vec<Message>, peer: &str, answer: &str) {
        // Add a tool result for the ask_peer call (the last assistant message)
        // Find the last ask_peer tool call ID
        if let Some(last_msg) = messages.iter().rev().find(|m| m.role == crate::schema::common::Role::Assistant) {
            if let Some(ref tool_calls) = last_msg.tool_calls {
                for tc in tool_calls {
                    if tc.name == "ask_peer" {
                        messages.push(Message::tool_result(&tc.id, answer));
                        return;
                    }
                }
            }
        }
        // Fallback: add as user message
        messages.push(Message::user(&format!("[Response from {}]: {}", peer, answer)));
    }

    /// Check if any tool call is `report_result` and extract its value.
    fn find_report_result(&self, calls: &[ToolCall]) -> Option<String> {
        for call in calls {
            if call.name == "report_result" {
                return Some(
                    call.arguments.get("result")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(empty report)")
                        .to_string()
                );
            }
        }
        None
    }

    /// Check if any tool call is `ask_peer` and extract (peer, question).
    fn find_ask_peer(&self, calls: &[ToolCall]) -> Option<(String, String)> {
        for call in calls {
            if call.name == "ask_peer" {
                let peer = call.arguments.get("peer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let question = call.arguments.get("question")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !peer.is_empty() && !question.is_empty() {
                    return Some((peer, question));
                }
            }
        }
        None
    }

    /// Execute a non-special tool call (not report_result or ask_peer).
    async fn execute_tool(&self, call: &ToolCall) -> String {
        match call.name.as_str() {
            "report_result" | "ask_peer" => String::new(), // handled elsewhere
            other => format!("Unknown tool: {}", other),
        }
    }

    fn select_provider(&self) -> Option<ModelProvider> {
        if let Some(ref p) = self.provider {
            return Some(p.clone());
        }
        let config = crate::schema::common::AppConfig::load().ok()?;
        let chat_group = config.group(crate::schema::common::ModelKind::Chat);
        chat_group.providers.iter().find(|p| p.enabled).cloned()
    }
}

// ── Builder ──

/// Builder for `CollaborativeAgent`.
#[derive(Default)]
pub struct CollaborativeAgentBuilder {
    name: Option<String>,
    role: Option<String>,
    system_prompt: Option<String>,
    capabilities: Vec<String>,
    description: Option<String>,
    provider: Option<ModelProvider>,
}

impl CollaborativeAgentBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    pub fn capabilities(mut self, caps: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.capabilities.extend(caps.into_iter().map(|c| c.into()));
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Build the agent, registering it with the message bus.
    pub fn build(
        self,
        adapter: Arc<dyn LlmAdapter>,
        bus: &Arc<MessageBus>,
    ) -> CollaborativeAgent {
        let name = self.name.expect("agent name is required");
        let role = self.role.unwrap_or_else(|| "general".to_string());
        let system_prompt = self.system_prompt.unwrap_or_default();

        // Create channel and register with bus
        let (sender, _receiver) = mpsc::unbounded_channel();
        bus.register_agent(name.clone(), sender);

        // Build contact book snapshot from current bus registrations
        let mut contact_book = ContactBook::new();
        for agent_name in bus.agent_names() {
            if agent_name != name {
                contact_book.add(
                    agent_name.to_string(),
                    "agent".to_string(),
                    vec![],
                    "Team member".to_string(),
                );
            }
        }

        CollaborativeAgent {
            name,
            role,
            system_prompt,
            adapter,
            contact_book,
            provider: self.provider,
            bus: bus.clone(),
        }
    }
}
