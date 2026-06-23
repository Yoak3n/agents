mod agent;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::provider::ProviderBalancer;
use crate::schema::common::{Message, ModelProvider, NullListener};

use super::subagent::{SubAgent, SubAgentContext, SubAgentResult, SubAgentStatus};
use crate::llm::{AgentResponse, LlmAdapter};

pub use agent::AgentStep;
pub use agent::{AgentContact, CollaborativeAgent, CollaborativeAgentBuilder, ContactBook};

/// Result from a team task execution.
pub struct TeamResult {
    pub task: String,
    pub agent_outputs: Vec<(String, String, String)>, // (name, role, output)
    pub synthesis: String,
}

/// Coordinates multiple agents to solve complex tasks.
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
    balancer: Option<ProviderBalancer>,
    agents: Vec<CollaborativeAgent>,
    max_rounds: usize,
}

impl TeamAgent {
    pub fn builder() -> TeamAgentBuilder {
        TeamAgentBuilder::default()
    }

    /// Select a provider using the balancer, falling back to the stored single provider.
    fn select_provider(&self) -> Option<ModelProvider> {
        if let Some(ref balancer) = self.balancer {
            balancer.select().cloned()
        } else {
            self.provider.clone()
        }
    }

    /// Execute a team task with full coordination (sequential).
    pub async fn execute_team_task(&self, task: &str) -> TeamResult {
        let conversation_id = uuid::Uuid::new_v4().to_string();
        let mut results = Vec::new();

        let agent_index: HashMap<String, usize> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name().to_string(), i))
            .collect();

        for agent in &self.agents {
            let name = agent.name().to_string();
            let role = agent.role().to_string();

            let mut messages = agent.build_task_messages(task);
            let mut final_output = String::new();

            for _round in 0..self.max_rounds {
                let provider = self.select_provider();
                match agent
                    .run_step_with_provider(&mut messages, provider.as_ref())
                    .await
                {
                    AgentStep::Done(result) => {
                        final_output = result;
                        break;
                    }
                    AgentStep::TextOutput(text) if text.is_empty() => {
                        continue;
                    }
                    AgentStep::TextOutput(text) => {
                        final_output = text;
                        break;
                    }
                    AgentStep::AskPeer { peer, question } => {
                        let answer = if let Some(&target_idx) = agent_index.get(&peer) {
                            let target = &self.agents[target_idx];
                            let mut visited = HashSet::new();
                            visited.insert(name.clone());
                            self.ask_agent(target, task, &question, &conversation_id, &mut visited)
                                .await
                        } else {
                            format!(
                                "[Error: agent '{}' not found. Available: {}]",
                                peer,
                                agent_index.keys().cloned().collect::<Vec<_>>().join(", ")
                            )
                        };

                        agent.inject_peer_answer(&mut messages, &peer, &answer);
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

    /// Execute a team task with all agents running concurrently.
    pub async fn execute_team_task_parallel(&self, task: &str) -> TeamResult {
        let agent_index: HashMap<String, usize> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name().to_string(), i))
            .collect();

        let balancer = self.balancer.clone();
        let fallback_provider = self.provider.clone();

        let futures: Vec<_> = self
            .agents
            .iter()
            .map(|agent| {
                let task = task.to_string();
                let conversation_id = uuid::Uuid::new_v4().to_string();
                let max_rounds = self.max_rounds;
                let agent_index = agent_index.clone();
                let agents = &self.agents;
                let balancer = balancer.clone();
                let fallback_provider = fallback_provider.clone();
                async move {
                    let name = agent.name().to_string();
                    let role = agent.role().to_string();

                    let mut messages = agent.build_task_messages(&task);
                    let mut final_output = String::new();

                    for _round in 0..max_rounds {
                        let provider = balancer
                            .as_ref()
                            .and_then(|b| b.select().cloned())
                            .or_else(|| fallback_provider.clone());
                        match agent
                            .run_step_with_provider(&mut messages, provider.as_ref())
                            .await
                        {
                            AgentStep::Done(result) => {
                                final_output = result;
                                break;
                            }
                            AgentStep::TextOutput(text) if text.is_empty() => continue,
                            AgentStep::TextOutput(text) => {
                                final_output = text;
                                break;
                            }
                            AgentStep::AskPeer { peer, question } => {
                                let answer = if let Some(&target_idx) = agent_index.get(&peer) {
                                    let target = &agents[target_idx];
                                    let mut visited = HashSet::new();
                                    visited.insert(name.clone());
                                    // Note: recursive ask_agent in parallel context
                                    // uses sequential sub-routes
                                    self_ask_agent_static(
                                        target,
                                        &task,
                                        &question,
                                        &conversation_id,
                                        &mut visited,
                                        agents,
                                        max_rounds,
                                    )
                                    .await
                                } else {
                                    format!(
                                        "[Error: agent '{}' not found. Available: {}]",
                                        peer,
                                        agent_index.keys().cloned().collect::<Vec<_>>().join(", ")
                                    )
                                };

                                agent.inject_peer_answer(&mut messages, &peer, &answer);
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

                    (name, role, final_output)
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
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
    /// Cycle detection prevents infinite recursion.
    async fn ask_agent(
        &self,
        agent: &CollaborativeAgent,
        original_task: &str,
        question: &str,
        conversation_id: &str,
        visited: &mut HashSet<String>,
    ) -> String {
        let context_msg = format!(
            "A teammate needs your help with a specific question related to the overall task.\n\n\
            Original task: {}\n\n\
            Their question: {}",
            original_task, question
        );

        let mut messages = agent.build_task_messages(&context_msg);

        let agent_index: HashMap<String, usize> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.name().to_string(), i))
            .collect();

        for _round in 0..self.max_rounds {
            let provider = self.select_provider();
            match agent
                .run_step_with_provider(&mut messages, provider.as_ref())
                .await
            {
                AgentStep::Done(result) => return result,
                AgentStep::TextOutput(text) if text.is_empty() => continue,
                AgentStep::TextOutput(text) => return text,
                AgentStep::AskPeer {
                    peer,
                    question: sub_q,
                } => {
                    // Cycle detection
                    if visited.contains(&peer) {
                        agent.inject_peer_answer(
                            &mut messages,
                            &peer,
                            &format!(
                                "[Error: circular ask_peer detected: {} -> {}]",
                                agent.name(),
                                peer
                            ),
                        );
                        continue;
                    }

                    let answer = if let Some(&idx) = agent_index.get(&peer) {
                        let target = &self.agents[idx];
                        visited.insert(peer.clone());
                        let result = Box::pin(self.ask_agent(
                            target,
                            original_task,
                            &sub_q,
                            conversation_id,
                            visited,
                        ))
                        .await;
                        visited.remove(&peer);
                        result
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

        let provider = match self.select_provider() {
            Some(p) => p,
            None => return summaries.join("\n\n"),
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
    providers: Vec<ModelProvider>,
}

impl Default for TeamAgentBuilder {
    fn default() -> Self {
        Self {
            agents: Vec::new(),
            max_rounds: 10,
            providers: Vec::new(),
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
        self.providers = vec![provider];
        self
    }

    pub fn providers(mut self, providers: Vec<ModelProvider>) -> Self {
        self.providers = providers;
        self
    }

    fn build_contact_books(&mut self) {
        let agent_info: Vec<(String, String, Vec<String>, String)> = self
            .agents
            .iter()
            .map(|a| {
                (
                    a.name().to_string(),
                    a.role().to_string(),
                    a.capabilities().to_vec(),
                    a.description_val().to_string(),
                )
            })
            .collect();

        for agent in &mut self.agents {
            let mut book = ContactBook::new();
            for (name, role, caps, desc) in &agent_info {
                if name != agent.name() {
                    book.add(name.clone(), role.clone(), caps.clone(), desc.clone());
                }
            }
            agent.set_contact_book(book);
        }
    }

    /// Build the team with a single provider (or none).
    pub fn build(mut self, adapter: Arc<dyn LlmAdapter>) -> TeamAgent {
        self.build_contact_books();

        TeamAgent {
            adapter,
            provider: self.providers.first().cloned(),
            balancer: None,
            agents: self.agents,
            max_rounds: self.max_rounds,
        }
    }

    /// Build the team with multiple providers and automatic load balancing.
    pub fn build_balanced(mut self, adapter: Arc<dyn LlmAdapter>) -> TeamAgent {
        self.build_contact_books();

        let balancer = if self.providers.len() > 1 {
            Some(ProviderBalancer::new(self.providers.clone()))
        } else {
            None
        };

        TeamAgent {
            adapter,
            provider: self.providers.first().cloned(),
            balancer,
            agents: self.agents,
            max_rounds: self.max_rounds,
        }
    }
}

// ── Static helper for parallel execution ──
// (avoids borrowing issues with async closures)

async fn self_ask_agent_static(
    agent: &CollaborativeAgent,
    original_task: &str,
    question: &str,
    conversation_id: &str,
    visited: &mut HashSet<String>,
    all_agents: &[CollaborativeAgent],
    max_rounds: usize,
) -> String {
    let context_msg = format!(
        "A teammate needs your help with a specific question related to the overall task.\n\n\
        Original task: {}\n\n\
        Their question: {}",
        original_task, question
    );

    let mut messages = agent.build_task_messages(&context_msg);

    let agent_index: HashMap<String, usize> = all_agents
        .iter()
        .enumerate()
        .map(|(i, a)| (a.name().to_string(), i))
        .collect();

    for _round in 0..max_rounds {
        match agent.run_step(&mut messages).await {
            AgentStep::Done(result) => return result,
            AgentStep::TextOutput(text) if text.is_empty() => continue,
            AgentStep::TextOutput(text) => return text,
            AgentStep::AskPeer {
                peer,
                question: sub_q,
            } => {
                if visited.contains(&peer) {
                    agent.inject_peer_answer(
                        &mut messages,
                        &peer,
                        &format!(
                            "[Error: circular ask_peer detected: {} -> {}]",
                            agent.name(),
                            peer
                        ),
                    );
                    continue;
                }

                let answer = if let Some(&idx) = agent_index.get(&peer) {
                    let target = &all_agents[idx];
                    visited.insert(peer.clone());
                    let result = Box::pin(self_ask_agent_static(
                        target,
                        original_task,
                        &sub_q,
                        conversation_id,
                        visited,
                        all_agents,
                        max_rounds,
                    ))
                    .await;
                    visited.remove(&peer);
                    result
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

// ── SubAgent impl ──

#[async_trait]
impl SubAgent for TeamAgent {
    fn name(&self) -> &str {
        "team"
    }

    fn description(&self) -> &str {
        "Coordinate multiple specialized agents to solve complex tasks"
    }

    fn capabilities(&self) -> Vec<String> {
        self.agents
            .iter()
            .flat_map(|a| a.capabilities().to_vec())
            .collect()
    }

    async fn execute(&self, input: &str, _ctx: SubAgentContext<'_>) -> SubAgentResult {
        let result = self.execute_team_task(input).await;

        SubAgentResult {
            output: result.synthesis,
            status: SubAgentStatus::Success,
            metadata: Some(json!({
                "agent_count": result.agent_outputs.len(),
                "agents": result.agent_outputs.iter()
                    .map(|(name, role, _)| format!("{} ({})", name, role))
                    .collect::<Vec<_>>()
            })),
        }
    }
}
