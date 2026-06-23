use yoakraft::prelude::*;

/// Demonstrates cost tracking with per-provider pricing.
///
/// Shows how to:
/// - Configure per-provider pricing (including cached input discounts)
/// - Query cost breakdown after a conversation
///
/// Usage:
///   cargo run -p yoakraft --example cost_tracking
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "demo",
        "https://api.openai.com/v1",
        "sk-your-api-key-here",
        "gpt-4o-mini",
    );

    // Configure pricing: GPT-4o-mini rates
    let pricing = PricingTable::new()
        .provider(
            "demo",
            PricingRule::new(0.00015, 0.0006).cached_input(0.000075),
        )
        .default(PricingRule::new(0.001, 0.002));

    let agent = CraftBuilder::new()
        .provider(provider)
        .cost_tracking(pricing)
        .build()?;

    // Run a few turns
    let questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain Rust ownership in one sentence.",
    ];

    for question in questions {
        println!("Q: {question}");
        let mut messages = vec![Message::user(question)];
        let reply = agent.run(&mut messages).await?;
        println!("A: {reply}\n");
    }

    // Query cost breakdown
    if let Some(tracker) = agent.cost_tracker() {
        println!("--- Cost Report ---");
        let total = tracker.estimated_cost().await;
        println!("Estimated total cost: ${:.6}", total);

        let breakdown = tracker.cost_by_provider().await;
        for (provider, cost) in &breakdown {
            println!("  {}: ${:.6}", provider, cost);
        }

        let usage = tracker.usage().await;
        for (provider, u) in &usage {
            println!(
                "  {} tokens: {} in ({} cached) / {} out ({} requests)",
                provider, u.input_tokens, u.cached_input_tokens, u.output_tokens, u.requests
            );
        }
    }

    Ok(())
}
