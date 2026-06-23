use std::io::{self, Write};

use yoakraft::prelude::*;

/// Full-featured multi-turn CLI using CraftAgent.
///
/// Demonstrates: memory, context management, cost tracking, file tools, skills.
///
/// Usage:
///   cargo run -p yoakraft --example craft_cli --features storage
///
/// Commands:
///   /quit      — exit
///   /clear     — clear conversation history
///   /cost      — show cost breakdown
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Configure provider ──
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "demo",
        "https://api.openai.com/v1",
        "sk-your-api-key-here",
        "gpt-4o-mini",
    );

    // ── 2. Configure pricing ──
    let pricing = PricingTable::new()
        .default(PricingRule::new(0.00015, 0.0006))
        .provider(
            "demo",
            PricingRule::new(0.00015, 0.0006).cached_input(0.000075),
        );

    // ── 3. Build CraftAgent with all features ──
    let agent = CraftBuilder::new()
        .provider(provider)
        .memory(MemoryConfig {
            max_injected: 3,
            auto_extract: true,
        })
        .context(ContextConfig { recent_to_keep: 20 })
        .cost_tracking(pricing)
        .file_tools(true)
        .max_rounds(10)
        .build()?;

    // ── 4. REPL ──
    let mut session = Session::with_system(
        "You are a helpful assistant with file system access. \
         Keep responses concise.",
    );

    println!("CraftAgent CLI (memory + context + cost + file tools)");
    println!("Commands: /quit, /clear, /cost\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        match input.as_str() {
            "/quit" | "/exit" => {
                println!("Goodbye!");
                break;
            }
            "/clear" => {
                session.clear();
                println!("(conversation cleared)\n");
                continue;
            }
            "/cost" => {
                if let Some(tracker) = agent.cost_tracker() {
                    let total = tracker.estimated_cost().await;
                    println!("(estimated cost: ${:.6})\n", total);
                } else {
                    println!("(cost tracking not enabled)\n");
                }
                continue;
            }
            _ => {}
        }

        session.add_user(&input);

        print!("Assistant: ");
        io::stdout().flush()?;

        let _reply = agent.run_in_session(&mut session).await?;
        println!();
    }

    Ok(())
}
