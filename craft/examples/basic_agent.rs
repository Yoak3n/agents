use yoakraft::prelude::*;

/// Minimal CraftAgent example — just a provider and one tool.
///
/// Usage:
///   cargo run -p yoakraft --example basic_agent
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "demo",
        "https://api.openai.com/v1",
        "sk-your-api-key-here",
        "gpt-4o-mini",
    );

    // Register a simple tool
    let mut tools = ToolRegistry::new();
    tools.register(
        ToolDefinition {
            name: "current_time".into(),
            description: "Get the current date and time.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        |_| {
            let now = chrono::Local::now();
            Ok(now.format("%Y-%m-%d %H:%M:%S").to_string())
        },
    );

    // Build the simplest possible CraftAgent
    let agent = CraftBuilder::new()
        .provider(provider)
        .tools(tools)
        .build()?;

    let mut messages = vec![Message::user("What time is it right now?")];

    let reply = agent.run(&mut messages).await?;
    println!("{reply}");

    Ok(())
}
