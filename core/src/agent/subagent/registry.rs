use std::collections::HashMap;
use std::sync::Arc;

use crate::schema::common::AppConfig;

use super::{SubAgent, SubAgentContext, SubAgentResult};

pub struct SubAgentRegistry {
    agents: HashMap<String, Arc<dyn SubAgent>>,
}

impl SubAgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register a sub-agent (wrapped in Arc internally).
    pub fn register(&mut self, agent: impl SubAgent + 'static) {
        let name = agent.name().to_string();
        self.agents.insert(name, Arc::new(agent));
    }

    /// Register a pre-built `Arc<dyn SubAgent>` (for sharing with hooks).
    pub fn register_arc(&mut self, agent: Arc<dyn SubAgent>) {
        let name = agent.name().to_string();
        self.agents.insert(name, agent);
    }

    /// Get an Arc reference to a registered sub-agent (for sharing with hooks).
    pub fn get_arc(&self, name: &str) -> Option<Arc<dyn SubAgent>> {
        self.agents.get(name).cloned()
    }

    pub fn get(&self, name: &str) -> Option<&dyn SubAgent> {
        self.agents.get(name).map(|a| a.as_ref())
    }

    /// List all registered agents as (name, description).
    pub fn all(&self) -> Vec<(&str, &str)> {
        self.agents
            .values()
            .map(|a| (a.name(), a.description()))
            .collect()
    }

    /// List all registered agents with their capabilities.
    pub fn list_capabilities(&self) -> HashMap<String, Vec<String>> {
        self.agents
            .values()
            .map(|a| (a.name().to_string(), a.capabilities()))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Execute a sub-agent by name.
    pub async fn execute(
        &self,
        name: &str,
        input: &str,
        context: SubAgentContext<'_>,
    ) -> Result<SubAgentResult, SubAgentError> {
        let agent = self
            .agents
            .get(name)
            .ok_or_else(|| SubAgentError::NotFound(name.to_string()))?;
        Ok(agent.execute(input, context).await)
    }

    /// Execute multiple sub-tasks sequentially.
    ///
    /// Each task is a `(agent_name, input)` pair. Results are returned in input order.
    pub async fn execute_sequential(
        &self,
        tasks: &[(&str, &str)],
        session_id: &str,
        app_config: &AppConfig,
    ) -> Vec<(String, Result<SubAgentResult, SubAgentError>)> {
        let mut results = Vec::with_capacity(tasks.len());
        for (name, input) in tasks {
            let context = SubAgentContext {
                session_id,
                message_history: &[],
                registry: Some(self),
                #[cfg(feature = "extension")]
                available_skills: &[],
                app_config,
            };
            let result = self.execute(name, input, context).await;
            results.push((name.to_string(), result));
        }
        results
    }

    /// Execute multiple sub-tasks concurrently.
    ///
    /// Each task is a `(agent_name, input)` pair. Results are returned in input order.
    /// Uses `futures::future::join_all` for concurrent execution.
    pub async fn execute_parallel(
        &self,
        tasks: &[(&str, &str)],
        session_id: &str,
        app_config: &AppConfig,
    ) -> Vec<(String, Result<SubAgentResult, SubAgentError>)> {
        let futures: Vec<_> = tasks
            .iter()
            .map(|(name, input)| {
                let agent = self.agents.get(*name).cloned();
                let name = name.to_string();
                let input = input.to_string();
                let session_id = session_id.to_string();
                let app_config_clone = app_config.clone();
                async move {
                    let agent = match agent {
                        Some(a) => a,
                        None => {
                            let err_name = name.clone();
                            return (name, Err(SubAgentError::NotFound(err_name)));
                        }
                    };
                    let context = SubAgentContext {
                        session_id: &session_id,
                        message_history: &[],
                        registry: None, // cannot share &self across threads easily
                        #[cfg(feature = "extension")]
                        available_skills: &[],
                        app_config: &app_config_clone,
                    };
                    let result = agent.execute(&input, context).await;
                    (name, Ok(result))
                }
            })
            .collect();

        futures::future::join_all(futures).await
    }
}

impl Default for SubAgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SubAgentError {
    #[error("sub-agent '{0}' not found")]
    NotFound(String),
}
