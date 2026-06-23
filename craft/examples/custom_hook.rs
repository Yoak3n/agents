use std::sync::Arc;

use async_trait::async_trait;
use yoakraft::prelude::*;

/// Demonstrates custom hooks with CraftAgent.
///
/// Two hooks are shown:
/// 1. A logging hook that prints every tool call
/// 2. A guard hook that blocks dangerous tool calls
///
/// Usage:
///   cargo run -p yoakraft --example custom_hook
// ── Hook 1: Tool call logger ──
struct ToolLogger;

#[async_trait]
impl AgentHook for ToolLogger {
    async fn before_tool_call(&self, _ctx: &HookContext<'_>, call: &ToolCall) -> HookResult {
        println!(
            "[ToolLogger] calling: {} with {}",
            call.name, call.arguments
        );
        HookResult::Continue
    }

    async fn after_tool_call(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        result: &mut String,
    ) -> HookResult {
        let preview = if result.len() > 100 {
            format!("{}...", &result[..100])
        } else {
            result.clone()
        };
        println!("[ToolLogger] {} -> {}", call.name, preview);
        HookResult::Continue
    }
}

// ── Hook 2: Tool guard ──

struct ToolGuard;

#[async_trait]
impl AgentHook for ToolGuard {
    async fn before_tool_call(&self, _ctx: &HookContext<'_>, call: &ToolCall) -> HookResult {
        // Block any tool call that tries to write to /etc or /sys
        let args_str = call.arguments.to_string();
        if args_str.contains("/etc") || args_str.contains("/sys") {
            return HookResult::Denied("access to system directories is not allowed".into());
        }
        HookResult::Continue
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "demo",
        "https://api.openai.com/v1",
        "sk-your-api-key-here",
        "gpt-4o-mini",
    );

    let agent = CraftBuilder::new()
        .provider(provider)
        .file_tools(true)
        .hook(ToolLogger)
        .hook_arc(Arc::new(ToolGuard))
        .build()?;

    let mut messages = vec![Message::user(
        "List the files in the current directory, then read Cargo.toml",
    )];

    let reply = agent.run(&mut messages).await?;
    println!("\nFinal reply:\n{reply}");

    Ok(())
}
