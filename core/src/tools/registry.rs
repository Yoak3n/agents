use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::schema::common::ToolDefinition;

use super::process::ProcessManager;

type ToolHandler = Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>;
type AsyncToolHandler = Box<
    dyn Fn(
            serde_json::Value,
            Arc<ProcessManager>,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

enum Handler {
    Sync(ToolHandler),
    Async(AsyncToolHandler),
}

pub struct ToolRegistry {
    tools: HashMap<String, (ToolDefinition, Handler)>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a synchronous tool handler.
    pub fn register(
        &mut self,
        definition: ToolDefinition,
        handler: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) {
        let name = definition.name.clone();
        self.tools
            .insert(name, (definition, Handler::Sync(Box::new(handler))));
    }

    /// Register an async tool handler that receives access to ProcessManager.
    pub fn register_async(
        &mut self,
        definition: ToolDefinition,
        handler: impl Fn(
            serde_json::Value,
            Arc<ProcessManager>,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync
        + 'static,
    ) {
        let name = definition.name.clone();
        self.tools
            .insert(name, (definition, Handler::Async(Box::new(handler))));
    }

    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name).map(|(def, _)| def)
    }

    /// Execute a tool by name. Dispatches to sync or async handler.
    pub async fn call(
        &self,
        name: &str,
        args: serde_json::Value,
        process_manager: &Arc<ProcessManager>,
    ) -> Result<String, String> {
        match self.tools.get(name) {
            Some((_, handler)) => match handler {
                Handler::Sync(h) => h(args),
                Handler::Async(h) => h(args, process_manager.clone()).await,
            },
            None => Err(format!("Tool '{name}' not found")),
        }
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|(def, _)| def.clone()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
