# agents

A Rust workspace for building AI agents — from low-level primitives to full-featured, production-ready agents.

## Crates

| Crate | Version | Description |
|-------|---------|-------------|
| [**yoakore**](./core) | 0.2.0 | Modular foundation — agent loop, LLM adapter, tools, hooks, multi-agent teams |
| [**yoakraft**](./craft) | 0.2.0 | Full-featured agent — adds memory, context management, cost tracking, skills |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                        │
├─────────────────────────────────────────────────────────┤
│  yoakraft (CraftBuilder → CraftAgent)                    │
│  ┌────────┐ ┌─────────┐ ┌──────┐ ┌───────┐             │
│  │ Memory │ │ Context │ │ Cost │ │ Skills│             │
│  │(own DB)│ └─────────┘ └──────┘ └───────┘             │
│  └────────┘                                              │
├─────────────────────────────────────────────────────────┤
│  yoakore (AgentBuilder → BaseAgent / TeamAgent / PlanAgent) │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌───────┐ ┌─────────┐    │
│  │ LLM  │ │Tools │ │ Hooks  │ │Events │ │ Storage │    │
│  └──────┘ └──────┘ └────────┘ └───────┘ └─────────┘    │
├─────────────────────────────────────────────────────────┤
│  Config │ Message │ Rate Limiting │ Embeddings            │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
# For a batteries-included agent (memory, context, cost tracking)
[dependencies]
yoakraft = "0.2"

# With built-in SQLite memory:
yoakraft = { version = "0.2", features = ["storage"] }

# Or for just the foundation (bring your own memory, etc.)
[dependencies]
yoakore = "0.2"
```

### Minimal example with yoakraft

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
        .context(ContextConfig::default())
        .cost_tracking(PricingTable::new().default(PricingRule::new(0.001, 0.002)))
        .file_tools(true)
        .build()?;

    let mut messages = vec![Message::user("Hello!")];
    let reply = agent.run(&mut messages).await?;
    println!("{reply}");
    Ok(())
}
```

With built-in SQLite memory (requires `storage` feature):

```rust
let agent = CraftBuilder::new()
    .provider(provider.clone())
    .memory(MemoryConfig::default())  // needs features = ["storage"]
    .context(ContextConfig::default())
    .build()?;
```

### Minimal example with yoakore

```rust
use yoakore::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "my-llm",
        "https://api.openai.com/v1",
        "sk-your-api-key",
        "gpt-4o-mini",
    );

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .with_on_event(|e| {
            if let AgentEvent::Delta(text) = e { print!("{text}"); }
        })
        .build_base();

    let mut messages = vec![Message::user("Hello!")];
    let reply = agent.execute(&provider, &mut messages).await?;
    println!("{reply}");
    Ok(())
}
```

## Examples

```bash
# yoakore examples
cargo run -p yoakore --example single_turn_cli
cargo run -p yoakore --example multi_turn_cli
cargo run -p yoakore --example message_injection

# yoakraft examples
cargo run -p yoakraft --example basic_agent
cargo run -p yoakraft --example craft_cli
cargo run -p yoakraft --example custom_hook
cargo run -p yoakraft --example cost_tracking
```

## License

MIT
