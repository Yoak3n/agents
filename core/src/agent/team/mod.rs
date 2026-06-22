mod agent;
mod bus;
mod message;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::schema::common::{Message, ModelProvider, NullListener};

use super::subagent::{SubAgent, SubAgentContext, SubAgentResult};
use crate::llm::{AgentResponse, LlmAdapter};

pub use agent::AgentStep;
pub use agent::{AgentContact, CollaborativeAgent, CollaborativeAgentBuilder, ContactBook};
pub use bus::MessageBus;
pub use message::{AgentMessage, AgentResponseMessage, MessageType};

/// Result from a team task execution.
pub struct TeamResult {
    pub task: String,
    pub agent_outputs: Vec<(String, String, String)>, // (name, role, output)
    pub synthesis: String,
}

/// Coordinates multiple agents to solve complex tasks via a shared message bus.
///
/// ## Architecture
///
/// Agents have two tools:
/// - `report_result` — the **only** way to return work
/// - `ask_peer` — request info from another specialist
///
/// The coordinator drives the loop:
/// 1. Send task to each agent
/// 2. Agent processes → may call `ask_peer`
/// 3. Coordinator routes the ask to the target agent
/// 4. Target agent answers → coordinator feeds answer back
/// 5. Repeat until agent calls `report_result`
/// 6. Coordinator synthesizes all results
///
/// ```text
///     coordinator          agent-A          agent-B
///         │                  │                │
///         │──task───────────▶│                │
///         │                  │                │
///         │◀─ask_peer(B)────│                │
///         │─────────────────────────────────▶│
///         │◀─report_result──────────────────│
///         │──answer─────────▶│                │
///         │                  │                │
///         │◀─report_result──│                │
///         │                  │                │
///         │─ synthesize(A, B) ─▶ final result│
/// ```
pub struct TeamAgent {
    adapter: Arc<dyn LlmAdapter>,
    provider: Option<ModelProvider>,
    #[allow(dead_code)]
    bus: Arc<MessageBus>,
    agents: Vec<CollaborativeAgent>,
    max_rounds: usize,
}

impl TeamAgent {
    pub fn builder() -> TeamAgentBuilder {
        TeamAgentBuilder::default()
    }

    /// Execute a team task with full coordination.
    pub async fn execute_team_task(&self, task: &str) -> TeamResult {
        let conversation_id = uuid::Uuid::new_v4().to_string();
        let mut results = Vec::new();

        // Build name→index lookup
        let agent_index: HashMap<String, usize> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name().to_string(), i))
            .collect();

        // Process each agent with coordinator-driven loop
        for agent in &self.agents {
            let name = agent.name().to_string();
            let role = agent.role().to_string();

            let mut messages = agent.build_task_messages(task);
            let mut final_output = String::new();

            for _round in 0..self.max_rounds {
                match agent.run_step(&mut messages).await {
                    AgentStep::Done(result) => {
                        final_output = result;
                        break;
                    }
                    AgentStep::TextOutput(text) if text.is_empty() => {
                        // Agent made tool calls (non-ask), loop continues
                        continue;
                    }
                    AgentStep::TextOutput(text) => {
                        // LLM returned text without using tools — treat as result
                        final_output = text;
                        break;
                    }
                    AgentStep::AskPeer { peer, question } => {
                        // Route the ask to the target agent
                        let answer = if let Some(&target_idx) = agent_index.get(&peer) {
                            let target = &self.agents[target_idx];
                            self.ask_agent(target, task, &question, &conversation_id)
                                .await
                        } else {
                            format!(
                                "[Error: agent '{}' not found. Available: {}]",
                                peer,
                                agent_index.keys().cloned().collect::<Vec<_>>().join(", ")
                            )
                        };

                        // Feed the answer back to the asking agent
                        agent.inject_peer_answer(&mut messages, &peer, &answer);
                        // Loop continues — agent will process the answer
                    }
                    AgentStep::Error(e) => {
                        final_output = format!("[{} error: {}]", name, e);
                        break;
                    }
                }
            }

            if final_output.is_empty() {
                final_output = format!("[{}: max rounds exceeded]", name);
            }

            results.push((name, role, final_output));
        }

        let synthesis = self.synthesize_results(task, &results).await;

        TeamResult {
            task: task.to_string(),
            agent_outputs: results,
            synthesis,
        }
    }

    /// Ask a target agent a question and get its answer.
    ///
    /// Runs a mini coordination loop: the target agent may itself ask other agents.
    async fn ask_agent(
        &self,
        agent: &CollaborativeAgent,
        original_task: &str,
        question: &str,
        conversation_id: &str,
    ) -> String {
        // Build a focused message for the target agent
        let context_msg = format!(
            "A teammate needs your help with a specific question related to the overall task.\n\n\
            Original task: {}\n\n\
            Their question: {}",
            original_task, question
        );

        let mut messages = agent.build_task_messages(&context_msg);

        // Build name→index for sub-routing
        let agent_index: HashMap<String, usize> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name().to_string(), i))
            .collect();

        for _round in 0..self.max_rounds {
            match agent.run_step(&mut messages).await {
                AgentStep::Done(result) => return result,
                AgentStep::TextOutput(text) if text.is_empty() => continue,
                AgentStep::TextOutput(text) => return text,
                AgentStep::AskPeer {
                    peer,
                    question: sub_q,
                } => {
                    // Recursive: this agent is asking another
                    let answer = if let Some(&idx) = agent_index.get(&peer) {
                        let target = &self.agents[idx];
                        Box::pin(self.ask_agent(target, original_task, &sub_q, conversation_id))
                            .await
                    } else {
                        format!("[Error: agent '{}' not found]", peer)
                    };
                    agent.inject_peer_answer(&mut messages, &peer, &answer);
                }
                AgentStep::Error(e) => return format!("[error: {}]", e),
            }
        }

        format!("[{}: max rounds exceeded answering question]", agent.name())
    }

    async fn synthesize_results(&self, task: &str, results: &[(String, String, String)]) -> String {
        let summaries: Vec<String> = results
            .iter()
            .map(|(name, role, output)| format!("[{} - {}]: {}", name, role, output))
            .collect();

        let prompt = format!(
            "Task: {}\n\nAgent contributions:\n{}\n\nPlease synthesize these contributions \
            into a coherent final response.",
            task,
            summaries.join("\n\n")
        );

        let provider = match self.provider.clone() {
            Some(p) => p,
            None => {
                // Fall back to config
                if let Ok(config) = crate::schema::common::AppConfig::load() {
                    let chat_group = config.group(crate::schema::common::ModelKind::Chat);
                    if let Some(p) = chat_group.providers.iter().find(|p| p.enabled).cloned() {
                        p
                    } else {
                        return summaries.join("\n\n");
                    }
                } else {
                    return summaries.join("\n\n");
                }
            }
        };
        let messages = vec![
            Message::system(
                "You are a team coordinator. Synthesize agent contributions into a coherent response.",
            ),
            Message::user(&prompt),
        ];

        match self
            .adapter
            .chat(&provider, &messages, &[], &NullListener)
            .await
        {
            Ok(AgentResponse::MessageComplete(msg)) => msg.content,
            _ => summaries.join("\n\n"),
        }
    }
}

// ── Builder ──

pub struct TeamAgentBuilder {
    agents: Vec<CollaborativeAgent>,
    max_rounds: usize,
    provider: Option<ModelProvider>,
}

impl Default for TeamAgentBuilder {
    fn default() -> Self {
        Self {
            agents: Vec::new(),
            max_rounds: 10,
            provider: None,
        }
    }
}

impl TeamAgentBuilder {
    pub fn add_agent(mut self, agent: CollaborativeAgent) -> Self {
        self.agents.push(agent);
        self
    }

    pub fn max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn build(self, adapter: Arc<dyn LlmAdapter>, bus: &Arc<MessageBus>) -> TeamAgent {
        TeamAgent {
            adapter,
            provider: self.provider,
            bus: bus.clone(),
            agents: self.agents,
            max_rounds: self.max_rounds,
        }
    }
}

// ── SubAgent impl ──

#[async_trait]
impl SubAgent for TeamAgent {
    fn name(&self) -> &str {
        "team"
    }

    fn description(&self) -> &str {
        "Coordinate multiple specialized agents to solve complex tasks via message passing"
    }

    async fn execute(&self, input: &str, _ctx: SubAgentContext<'_>) -> SubAgentResult {
        let result = self.execute_team_task(input).await;

        SubAgentResult {
            output: result.synthesis,
            metadata: Some(json!({
                "agent_count": result.agent_outputs.len(),
                "agents": result.agent_outputs.iter()
                    .map(|(name, role, _)| format!("{} ({})", name, role))
                    .collect::<Vec<_>>()
            })),
        }
    }
}
