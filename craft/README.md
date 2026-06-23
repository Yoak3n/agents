# yoakraft

A full-featured AI agent built on [yoakore](https://crates.io/crates/yoakore) — batteries included with memory, context management, cost tracking, skills, and built-in tools.

## Features

- **Memory** — persistent memory with keyword retrieval and automatic extraction from assistant replies
- **Context management** — sliding-window compression to stay within token limits
- **Cost tracking** — per-provider token usage recording and cost estimation with cached-input discounts
- **Skills** — load SKILL.md files from a workspace and inject matching skills into prompts
- **Built-in tools** — `read_file`, `write_file`, `list_directory`, `search_files` out of the box
- **Composable hooks** — all features are implemented as `AgentHook`s and can be mixed, replaced, or extended

## Quick Start

```toml
[dependencies]
yoakraft = "0.1"
```

```rust
use yoakraft::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "my-llm",
        "https://api.openai.com/v1",
        "sk-your-api-key",
        "gpt-4o-mini",
    );

    let agent = CraftBuilder::new()
        .provider(provider.clone())
        .memory(MemoryConfig::default())
        .context(ContextConfig::default())
        .cost_tracking(
            PricingTable::new()
                .provider("my-llm", PricingRule::new(0.00015, 0.0006))
        )
        .file_tools(true)
        .build()?;

    let mut messages = vec![Message::user("Hello!")];
    let reply = agent.run(&mut messages).await?;
    println!("{reply}");
    Ok(())
}
```

## Builder API

`CraftBuilder` provides a fluent API with two customization patterns:

### Convenience methods (recommended)

```rust
let agent = CraftBuilder::new()
    .provider(provider)
    .memory(MemoryConfig { max_injected: 5, auto_extract: true })
    .context(ContextConfig { recent_to_keep: 20 })
    .cost_tracking(PricingTable::new().default(PricingRule::new(0.001, 0.002)))
    .file_tools(true)
    .skills("./workspace")     // load SKILL.md files
    .max_rounds(10)
    .build()?;
```

### Trait-object replacement

Bring your own implementations of `MemoryProvider`, `ContextManager`, or `CostCalculator`:

```rust
let agent = CraftBuilder::new()
    .provider(provider)
    .memory_provider(Arc::new(MyCustomMemory))
    .context_manager(Arc::new(MyCustomContext))
    .cost_calculator(Arc::new(MyCustomCostCalc))
    .build()?;
```

### Custom hooks

Append any `AgentHook` implementation:

```rust
let agent = CraftBuilder::new()
    .provider(provider)
    .hook(MyLoggingHook)
    .hook_arc(Arc::new(MyGuardHook))
    .build()?;
```

## Components

### Memory

`DefaultMemory` uses the SQLite-backed `Storage` for persistence. It:
- **Retrieves** relevant memories via keyword search before each LLM call
- **Extracts** new memories from assistant replies (when `auto_extract` is enabled)

```rust
.memory(MemoryConfig {
    max_injected: 3,    // max memories injected per call
    auto_extract: true,  // auto-extract from replies
})
```

### Context Management

`DefaultContext` uses a sliding-window strategy:
- When token usage exceeds 80% of the context window, older messages are replaced with a summary
- System messages and the N most recent messages are always preserved

```rust
.context(ContextConfig { recent_to_keep: 20 })
```

### Cost Tracking

`CostTracker` records per-provider token usage and estimates cost:

```rust
let pricing = PricingTable::new()
    .provider("gpt-4o", PricingRule::new(0.0025, 0.01).cached_input(0.00125))
    .provider("deepseek", PricingRule::new(0.00014, 0.00028))
    .default(PricingRule::new(0.001, 0.002));

let agent = CraftBuilder::new()
    .provider(provider)
    .cost_tracking(pricing)
    .build()?;

// After running:
if let Some(tracker) = agent.cost_tracker() {
    println!("Cost: ${:.6}", tracker.estimated_cost().await);
}
```

### Skills

Load `SKILL.md` files from a workspace directory. Matching skills are automatically injected into the system prompt:

```rust
CraftBuilder::new()
    .provider(provider)
    .skills("./my-workspace")  // loads skills from ./my-workspace/skills/
    .build()?;
```

## Running Examples

```bash
cargo run -p yoakraft --example basic_agent
cargo run -p yoakraft --example multi_turn_cli
cargo run -p yoakraft --example custom_hook
cargo run -p yoakraft --example cost_tracking
```

## License

MIT
