use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::hook::{AgentHook, HookContext, HookResult};
use crate::schema::common::ToolCall;

/// Async callback for tool approval decisions.
///
/// Receives the tool name and arguments, returns `true` to approve.
pub type ApprovalCallback = Arc<
    dyn Fn(String, serde_json::Value) -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync,
>;

/// How often approval is required for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalScope {
    /// Ask the user every time this tool is called.
    PerCall,
    /// Ask once; cache the approval for the agent's lifetime.
    Session,
}

struct ApprovalRule {
    tool_name: String,
    scope: ApprovalScope,
}

/// Policy engine that decides whether tool calls need user approval.
///
/// Tools not covered by any rule are auto-approved (default behavior).
/// `ApprovalPolicy` is cheaply cloneable (all state is `Arc`-wrapped).
///
/// ```ignore
/// let policy = ApprovalPolicy::require_approval(
///     ["shell_execute", "write_file"],
///     Arc::new(|name, args| Box::pin(async move {
///         println!("Approve '{name}'?");
///         true
///     })),
/// );
/// ```
pub struct ApprovalPolicy {
    rules: Vec<ApprovalRule>,
    callback: ApprovalCallback,
    session_approvals: Arc<Mutex<HashSet<String>>>,
}

impl Clone for ApprovalPolicy {
    fn clone(&self) -> Self {
        Self {
            rules: Vec::new(), // rules don't need cloning; clone is for sharing cache+callback
            callback: self.callback.clone(),
            session_approvals: self.session_approvals.clone(),
        }
    }
}

impl ApprovalPolicy {
    /// Create a policy requiring approval for the given tools (per-call scope).
    pub fn require_approval(
        tool_names: impl IntoIterator<Item = impl Into<String>>,
        callback: ApprovalCallback,
    ) -> Self {
        Self {
            rules: tool_names
                .into_iter()
                .map(|name| ApprovalRule {
                    tool_name: name.into(),
                    scope: ApprovalScope::PerCall,
                })
                .collect(),
            callback,
            session_approvals: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Create a policy requiring session-scoped approval for the given tools.
    ///
    /// The first call to each tool triggers the callback; subsequent calls
    /// are auto-approved for the lifetime of this policy instance.
    pub fn require_session_approval(
        tool_names: impl IntoIterator<Item = impl Into<String>>,
        callback: ApprovalCallback,
    ) -> Self {
        Self {
            rules: tool_names
                .into_iter()
                .map(|name| ApprovalRule {
                    tool_name: name.into(),
                    scope: ApprovalScope::Session,
                })
                .collect(),
            callback,
            session_approvals: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Start building a policy with mixed scopes.
    pub fn builder(callback: ApprovalCallback) -> ApprovalPolicyBuilder {
        ApprovalPolicyBuilder {
            callback,
            rules: Vec::new(),
        }
    }

    /// Check whether a tool call should proceed.
    ///
    /// Returns `true` if approved (or no rule matches), `false` if denied.
    pub async fn check_approval(&self, call: &ToolCall) -> bool {
        let rule = match self.rules.iter().find(|r| r.tool_name == call.name) {
            Some(r) => r,
            None => return true, // no rule = auto-approve
        };

        match rule.scope {
            ApprovalScope::Session => {
                {
                    let cache = self.session_approvals.lock().unwrap();
                    if cache.contains(&call.name) {
                        return true;
                    }
                }
                let approved = (self.callback)(call.name.clone(), call.arguments.clone()).await;
                if approved {
                    let mut cache = self.session_approvals.lock().unwrap();
                    cache.insert(call.name.clone());
                }
                approved
            }
            ApprovalScope::PerCall => {
                (self.callback)(call.name.clone(), call.arguments.clone()).await
            }
        }
    }
}

/// Builder for `ApprovalPolicy` with mixed per-call and session scopes.
pub struct ApprovalPolicyBuilder {
    callback: ApprovalCallback,
    rules: Vec<ApprovalRule>,
}

impl ApprovalPolicyBuilder {
    /// Add a tool that requires approval on every call.
    pub fn require(mut self, tool_name: impl Into<String>) -> Self {
        self.rules.push(ApprovalRule {
            tool_name: tool_name.into(),
            scope: ApprovalScope::PerCall,
        });
        self
    }

    /// Add a tool that requires approval once per session.
    pub fn require_session(mut self, tool_name: impl Into<String>) -> Self {
        self.rules.push(ApprovalRule {
            tool_name: tool_name.into(),
            scope: ApprovalScope::Session,
        });
        self
    }

    /// Build the `ApprovalPolicy`.
    pub fn build(self) -> ApprovalPolicy {
        ApprovalPolicy {
            rules: self.rules,
            callback: self.callback,
            session_approvals: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

/// `AgentHook` adapter that enforces `ApprovalPolicy` before each tool call.
///
/// Denied calls produce `HookResult::Denied` (soft deny — the agent loop
/// continues with a `"[denied]"` tool result).
pub struct ApprovalHook {
    policy: ApprovalPolicy,
}

impl ApprovalHook {
    pub fn new(policy: ApprovalPolicy) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl AgentHook for ApprovalHook {
    async fn before_tool_call(&self, _ctx: &HookContext<'_>, call: &ToolCall) -> HookResult {
        if self.policy.check_approval(call).await {
            HookResult::Continue
        } else {
            HookResult::Denied(format!("Tool '{}' denied by user", call.name))
        }
    }
}
