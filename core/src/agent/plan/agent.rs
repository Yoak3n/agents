use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::agent::subagent::{SubAgent, SubAgentContext, SubAgentResult, SubAgentStatus};
use crate::error::AgentError;
use crate::llm::{AgentResponse, LlmAdapter};
use crate::provider::ProviderBalancer;
use crate::schema::common::{AppConfig, Message, ModelProvider, NullListener};

/// A single step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Unique identifier within the plan (e.g. "t1", "t2").
    pub id: String,
    /// Name of the sub-agent to execute this task.
    pub agent_name: String,
    /// Input prompt for the sub-agent.
    pub input: String,
    /// IDs of subtasks that must complete before this one starts.
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// Result from executing a plan.
pub struct PlanResult {
    /// Original task description.
    pub task: String,
    /// The generated plan (list of subtasks).
    pub plan: Vec<Subtask>,
    /// Results from each subtask: (subtask_id, agent_name, output).
    pub subtask_results: Vec<(String, String, String)>,
    /// Final synthesized output.
    pub synthesis: String,
}

/// An agent that decomposes complex tasks into a plan and delegates
/// to sub-agents via the `SubAgentRegistry`.
///
/// ## Workflow
///
/// 1. **Plan**: LLM analyzes the task and available sub-agents, produces a
///    JSON plan with subtasks and dependency relationships.
/// 2. **Execute**: Subtasks with satisfied dependencies run in order.
///    Independent subtasks at the same level run sequentially (registry
///    holds `&dyn SubAgent` which prevents easy parallel send).
/// 3. **Synthesize**: LLM combines all subtask results into a coherent response.
///
/// ```text
///     task → LLM(plan) → resolve deps → execute subtasks → LLM(synthesize) → result
/// ```
///
/// PlanAgent itself implements `SubAgent`, so it can be nested inside
/// other PlanAgents or used in a SubAgentRegistry.
pub struct PlanAgent {
    adapter: Arc<dyn LlmAdapter>,
    provider: Option<ModelProvider>,
    balancer: Option<ProviderBalancer>,
    max_plan_rounds: usize,
    app_config: Option<AppConfig>,
}

impl PlanAgent {
    pub fn new(adapter: Arc<dyn LlmAdapter>) -> Self {
        Self {
            adapter,
            provider: None,
            balancer: None,
            max_plan_rounds: 10,
            app_config: None,
        }
    }

    pub fn provider(mut self, provider: ModelProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set multiple providers with automatic load balancing.
    pub fn providers(mut self, providers: Vec<ModelProvider>) -> Self {
        if providers.len() > 1 {
            self.balancer = Some(ProviderBalancer::new(providers.clone()));
        }
        self.provider = providers.into_iter().next();
        self
    }

    pub fn max_plan_rounds(mut self, rounds: usize) -> Self {
        self.max_plan_rounds = rounds;
        self
    }

    pub fn app_config(mut self, config: AppConfig) -> Self {
        self.app_config = Some(config);
        self
    }

    pub(crate) fn app_config_opt(mut self, config: Option<AppConfig>) -> Self {
        self.app_config = config;
        self
    }

    /// Execute a task by planning, running subtasks, and synthesizing.
    pub async fn execute_plan(
        &self,
        task: &str,
        registry: &crate::agent::subagent::SubAgentRegistry,
    ) -> Result<PlanResult, AgentError> {
        let session_id = uuid::Uuid::new_v4().to_string();

        // Step 1: Generate plan
        let plan = self.generate_plan(task, registry).await?;

        // Step 2: Execute subtasks with dependency resolution
        let subtask_results = self.execute_subtasks(&plan, registry, &session_id).await?;

        // Step 3: Synthesize results
        let synthesis = self.synthesize(task, &plan, &subtask_results).await;

        Ok(PlanResult {
            task: task.to_string(),
            plan,
            subtask_results,
            synthesis,
        })
    }

    /// Use LLM to generate a plan from the task and available agents.
    async fn generate_plan(
        &self,
        task: &str,
        registry: &crate::agent::subagent::SubAgentRegistry,
    ) -> Result<Vec<Subtask>, AgentError> {
        let agent_descs: Vec<String> = registry
            .all()
            .iter()
            .map(|(name, desc)| {
                let caps = registry
                    .list_capabilities()
                    .get(*name)
                    .map(|c| format!(" [{}]", c.join(", ")))
                    .unwrap_or_default();
                format!("- {}{}: {}", name, caps, desc)
            })
            .collect();

        let prompt = format!(
            "You are a task planner. Given a complex task and available sub-agents, \
            decompose the task into a sequence of subtasks.\n\n\
            ## Available agents\n{}\n\n\
            ## Task\n{}\n\n\
            ## Output format\n\
            Return ONLY a JSON array of subtasks. Each subtask has:\n\
            - `id`: unique identifier (e.g. \"t1\", \"t2\")\n\
            - `agent_name`: which agent to use (must match an available agent)\n\
            - `input`: specific instructions for this sub-agent\n\
            - `depends_on`: array of subtask IDs that must complete first (empty if independent)\n\n\
            Example:\n\
            [\n\
              {{\"id\": \"t1\", \"agent_name\": \"search\", \"input\": \"Find info about X\", \"depends_on\": []}},\n\
              {{\"id\": \"t2\", \"agent_name\": \"writer\", \"input\": \"Write summary\", \"depends_on\": [\"t1\"]}}\n\
            ]\n\n\
            Use `{{{{tN.result}}}}` in input to reference a previous subtask's output.\n\
            Keep the plan simple and focused. 2-5 subtasks is usually enough.",
            agent_descs.join("\n"),
            task,
        );

        let messages = vec![Message::user(&prompt)];
        let provider = self.select_provider()?;

        let response = self
            .adapter
            .chat(&provider, &messages, &[], &NullListener)
            .await?;

        let content = match response {
            AgentResponse::MessageComplete(msg) => msg.content,
            AgentResponse::ToolCalls(_) => {
                return Err(AgentError::Other(
                    "plan generation returned tool calls".to_string(),
                ));
            }
        };

        parse_plan(&content)
            .map_err(|e| AgentError::Other(format!("failed to parse plan: {e}\nRaw: {content}")))
    }

    /// Execute subtasks respecting dependency order.
    ///
    /// Subtasks whose dependencies are all satisfied are executed in sequence.
    /// `{{tN.result}}` placeholders are resolved from completed results.
    async fn execute_subtasks(
        &self,
        plan: &[Subtask],
        registry: &crate::agent::subagent::SubAgentRegistry,
        session_id: &str,
    ) -> Result<Vec<(String, String, String)>, AgentError> {
        let app_config = match self.app_config.clone() {
            Some(c) => c,
            None => AppConfig::load().map_err(|e| AgentError::Other(e.to_string()))?,
        };

        let mut completed: HashMap<String, String> = HashMap::new();
        let mut all_results: Vec<(String, String, String)> = Vec::new();
        let mut remaining: HashSet<String> = plan.iter().map(|s| s.id.clone()).collect();

        for _round in 0..self.max_plan_rounds {
            if remaining.is_empty() {
                break;
            }

            // Find subtasks whose deps are all satisfied
            let ready: Vec<&Subtask> = plan
                .iter()
                .filter(|s| remaining.contains(&s.id))
                .filter(|s| s.depends_on.iter().all(|d| completed.contains_key(d)))
                .collect();

            if ready.is_empty() {
                return Err(AgentError::Other(format!(
                    "plan deadlock: remaining subtasks have unsatisfied dependencies. \
                     Completed: {:?}, Remaining: {:?}",
                    completed.keys().collect::<Vec<_>>(),
                    remaining.iter().collect::<Vec<_>>(),
                )));
            }

            for subtask in &ready {
                let resolved_input = resolve_placeholders(&subtask.input, &completed);
                let context = SubAgentContext {
                    session_id,
                    message_history: &[],
                    registry: Some(registry),
                    #[cfg(feature = "extension")]
                    available_skills: &[],
                    app_config: &app_config,
                };

                match registry.get(&subtask.agent_name) {
                    Some(agent) => {
                        let result = agent.execute(&resolved_input, context).await;
                        let output = result.output.clone();
                        completed.insert(subtask.id.clone(), output.clone());
                        all_results.push((subtask.id.clone(), subtask.agent_name.clone(), output));
                    }
                    None => {
                        let err = format!("agent '{}' not found", subtask.agent_name);
                        completed.insert(subtask.id.clone(), err.clone());
                        all_results.push((subtask.id.clone(), subtask.agent_name.clone(), err));
                    }
                }
                remaining.remove(&subtask.id);
            }
        }

        Ok(all_results)
    }

    /// Synthesize all subtask results into a final response.
    async fn synthesize(
        &self,
        task: &str,
        plan: &[Subtask],
        results: &[(String, String, String)],
    ) -> String {
        let summaries: Vec<String> = results
            .iter()
            .map(|(id, agent, output)| {
                let subtask_desc = plan
                    .iter()
                    .find(|s| &s.id == id)
                    .map(|s| s.input.as_str())
                    .unwrap_or("");
                format!(
                    "[{} - {}] Task: {}\nOutput: {}",
                    id, agent, subtask_desc, output
                )
            })
            .collect();

        let prompt = format!(
            "Task: {}\n\n\
            Plan execution results:\n{}\n\n\
            Synthesize these results into a coherent final response. \
            Focus on completeness and accuracy.",
            task,
            summaries.join("\n\n"),
        );

        let provider = match self.select_provider() {
            Ok(p) => p,
            Err(_) => return summaries.join("\n\n"),
        };

        let messages = vec![
            Message::system(
                "You are a synthesis agent. Combine subtask results into a coherent response.",
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

    fn select_provider(&self) -> Result<ModelProvider, AgentError> {
        if let Some(ref balancer) = self.balancer {
            return balancer
                .select()
                .cloned()
                .ok_or_else(|| AgentError::Other("no enabled provider in balancer".to_string()));
        }
        if let Some(ref p) = self.provider {
            return Ok(p.clone());
        }
        let config = AppConfig::load().map_err(|e| AgentError::Other(e.to_string()))?;
        let chat_group = config.group(crate::schema::common::ModelKind::Chat);
        chat_group
            .providers
            .iter()
            .find(|p| p.enabled)
            .cloned()
            .ok_or_else(|| AgentError::Other("no enabled chat provider".to_string()))
    }
}

/// Replace `{{tN.result}}` placeholders with actual results.
fn resolve_placeholders(input: &str, results: &HashMap<String, String>) -> String {
    let mut output = input.to_string();
    for (id, result) in results {
        let placeholder = format!("{{{{{id}.result}}}}");
        output = output.replace(&placeholder, result);
    }
    output
}

/// Parse the LLM's JSON output into a list of subtasks.
fn parse_plan(raw: &str) -> Result<Vec<Subtask>, String> {
    // Try to extract JSON array from the response (LLM may wrap it in markdown)
    let json_str = if let Some(start) = raw.find('[') {
        if let Some(end) = raw.rfind(']') {
            &raw[start..=end]
        } else {
            raw
        }
    } else {
        raw
    };

    serde_json::from_str::<Vec<Subtask>>(json_str).map_err(|e| format!("JSON parse error: {e}"))
}

// ── SubAgent impl ──

#[async_trait]
impl SubAgent for PlanAgent {
    fn name(&self) -> &str {
        "plan"
    }

    fn description(&self) -> &str {
        "Decompose complex tasks into a plan and delegate to specialized sub-agents"
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["planning".to_string(), "decomposition".to_string()]
    }

    async fn execute(&self, input: &str, ctx: SubAgentContext<'_>) -> SubAgentResult {
        let registry = match ctx.registry {
            Some(r) => r,
            None => {
                return SubAgentResult::failed("PlanAgent requires a SubAgentRegistry in context");
            }
        };

        match self.execute_plan(input, registry).await {
            Ok(result) => SubAgentResult {
                output: result.synthesis,
                status: SubAgentStatus::Success,
                metadata: Some(json!({
                    "subtask_count": result.subtask_results.len(),
                    "subtasks": result.subtask_results.iter()
                        .map(|(id, agent, _)| format!("{} ({})", id, agent))
                        .collect::<Vec<_>>()
                })),
            },
            Err(e) => SubAgentResult::failed(e.to_string()),
        }
    }
}

// ── Builder ──

pub struct PlanAgentBuilder {
    adapter: Option<Arc<dyn LlmAdapter>>,
    providers: Vec<ModelProvider>,
    max_plan_rounds: usize,
    app_config: Option<AppConfig>,
}

impl Default for PlanAgentBuilder {
    fn default() -> Self {
        Self {
            adapter: None,
            providers: Vec::new(),
            max_plan_rounds: 10,
            app_config: None,
        }
    }
}

impl PlanAgentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn adapter(mut self, adapter: Arc<dyn LlmAdapter>) -> Self {
        self.adapter = Some(adapter);
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

    pub fn max_plan_rounds(mut self, rounds: usize) -> Self {
        self.max_plan_rounds = rounds;
        self
    }

    pub fn app_config(mut self, config: AppConfig) -> Self {
        self.app_config = Some(config);
        self
    }

    pub fn build(self) -> PlanAgent {
        let adapter = self.adapter.expect("adapter is required");
        PlanAgent::new(adapter)
            .providers(self.providers)
            .max_plan_rounds(self.max_plan_rounds)
            .app_config_opt(self.app_config)
    }
}
