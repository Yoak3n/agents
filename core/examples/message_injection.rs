use std::io::{self, Write};
use std::sync::Arc;

use serde_json::json;

use yoakore::prelude::*;

/// Message injection demo — send new messages while the agent is running.
///
/// While the agent is processing (e.g. calling tools in a loop), type a
/// new message in the terminal. It will be injected into the conversation
/// after the current tool-call round finishes, before the next LLM call.
///
/// Usage:
///   cargo run -p yoakore --example message_injection
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Provider ──
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "inject-demo",
        "https://api.openai.com/v1",
        "sk-your-api-key-here",
        "gpt-4o-mini",
    );

    // ── 2. Tools ──
    let mut tools = ToolRegistry::new();

    // A slow tool — simulates work that takes time, giving the user
    // a window to inject messages.
    tools.register(
        ToolDefinition {
            name: "slow_search".into(),
            description: "Search the knowledge base (takes a few seconds)".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "search query" }
                },
                "required": ["query"]
            }),
        },
        |args| {
            let query = args["query"].as_str().unwrap_or("");
            // In a real app this would be an async DB call
            Ok(format!(
                "Search results for '{}': [result1, result2, result3]",
                query
            ))
        },
    );

    // ── 3. Create inject channel ──
    let (inject_tx, inject_rx) = BaseAgent::inject_channel(16);

    // ── 4. Build agent with inject channel ──
    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .max_rounds(10)
        .tools(Arc::new(tools))
        .inject_channel(inject_rx)
        .with_on_event(|event| match event {
            AgentEvent::Delta(text) => print!("{text}"),
            AgentEvent::ThinkingDelta(text) => eprint!("{text}"),
            AgentEvent::Thinking => eprint!("\n[thinking] "),
            AgentEvent::ToolCallStart(tc) => {
                eprint!("\n[tool: {}] ", tc.name);
            }
            AgentEvent::ToolCallResult { result, .. } => {
                eprintln!(" -> {}", result.chars().take(80).collect::<String>());
            }
            AgentEvent::Done => eprintln!("\n[done]"),
            _ => {}
        })
        .build_base();

    // ── 5. Spawn stdin reader task ──
    // This task reads lines from stdin and sends them through the inject channel.
    // The agent will pick them up after the current tool-call round.
    let inject_handle = tokio::spawn(async move {
        eprintln!("[inject] Type messages while the agent runs. Press Ctrl+C to stop.");
        loop {
            let line = tokio::task::spawn_blocking(|| {
                eprint!("\n[inject] > ");
                io::stderr().flush().ok();
                let mut buf = String::new();
                io::stdin().read_line(&mut buf).ok();
                buf.trim().to_string()
            })
            .await
            .unwrap_or_default();

            if line.is_empty() {
                continue;
            }

            if inject_tx.send(Message::user(&line)).await.is_err() {
                break; // agent finished, channel closed
            }
            eprintln!("[inject] queued: {}", line);
        }
    });

    // ── 6. Run agent ──
    let mut messages = vec![
        Message::system(
            "You are a research assistant. Use the slow_search tool to look up \
             information. After each search, summarize what you found and ask if \
             the user wants to refine the search.",
        ),
        Message::user("Search for information about Rust async programming"),
    ];

    println!("Assistant:");
    let reply = agent.execute(&provider, &mut messages).await?;

    if !reply.is_empty() {
        println!("\n\nFinal reply: {}", reply);
    }

    // ── 7. Cleanup ──
    inject_handle.abort();

    // Show what messages ended up in the conversation
    println!(
        "\n--- Conversation history ({} messages) ---",
        messages.len()
    );
    for (i, msg) in messages.iter().enumerate() {
        let preview: String = msg.content.chars().take(80).collect();
        println!("  [{}] {:?}: {}", i, msg.role, preview);
    }

    Ok(())
}
