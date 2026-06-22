use std::collections::HashMap;
use std::sync::Arc;

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

    /// Register a pre-built Arc<dyn SubAgent> (for sharing with hooks).
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
