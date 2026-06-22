# yoakore

A modular, extensible foundation for building AI agents in Rust.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Your Application                  │
├─────────────────────────────────────────────────────┤
│  BaseAgent       │  TeamAgent / CollaborativeAgent  │
│  (single agent)  │  (multi-agent coordination)      │
├─────────────────────────────────────────────────────┤
│  AgentLike trait │  AgentHook trait │ EventListener  │
├─────────────────────────────────────────────────────┤
│  LlmAdapter trait  (OpenAI-compatible HTTP)         │
├─────────────────────────────────────────────────────┤
│  ToolRegistry │ ProcessManager │ SubAgentRegistry   │
├─────────────────────────────────────────────────────┤
│  Config │ Message │ ToolCall │ AgentEvent │ State    │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use yoakore::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    // 1. Load config (from config.json)
    let config = AppConfig::load()?;
    let provider = config.chat.providers.into_iter()
        .find(|p| p.enabled)
        .expect("no chat provider");

    // 2. Create adapter and agent
    let adapter = OpenAIAdapter::new();
    let agent = BaseAgent::new(adapter)
        .with_max_rounds(10)
        .with_on_event(|event| {
            if let AgentEvent::Delta(text) = event {
                print!("{}", text);
            }
        });

    // 3. Run
    let mut messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ];
    let reply = agent.execute(&provider, &mut messages).await?;
    println!("\n{}", reply);
    Ok(())
}
```

## Configuration

`config.json` at the working directory:

```json
{
  "chat": {
    "providers": [{
      "id": "openai",
      "kind": "chat",
      "name": "GPT-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-...",
      "model": "gpt-4o",
      "max_output": 4096,
      "weight": 1,
      "requests_per_minute": 60,
      "tier": 3,
      "enabled": true,
      "style": "openai",
      "thinking": { "type": "auto" }
    }]
  },
  "embedding": { "providers": [] }
}
```

### API Style

The `style` field controls request format:

| Style | Providers | Thinking support |
|-------|-----------|-----------------|
| `openai` (default) | OpenAI, DeepSeek, most compatible APIs | `thinking` (toggle) + `reasoning_effort` (intensity) |
| `anthropic` | Claude via Messages API | `thinking` (combined toggle + budget) |

### Thinking / Reasoning Mode

The `thinking` field controls deep reasoning. When set to `default`, no explicit config is sent — the API uses its own default (thinking is ON for o1/o3/Claude/DeepSeek etc.).

```json
{ "type": "default" }                           // 不显式配置，由 API 决定
{ "type": "disabled" }                          // 显式关闭思考
{ "type": "auto" }                              // 显式启用，中等强度
{ "type": "effort", "level": "low" }            // low / medium / high / max
{ "type": "effort", "level": "max" }            // DeepSeek 等支持的最高强度
{ "type": "budget", "tokens": 10000 }           // token 预算（Anthropic 风格）
```

Cross-API mapping is automatic — effort levels map to approximate budgets and vice versa.

For DeepSeek, thinking is enabled by default. Use `disabled` to turn it off, or `effort` with `max` for maximum reasoning depth.

## Core Concepts

### BaseAgent

The foundation agent with a built-in tool-use loop:

```rust
let agent = BaseAgent::new(adapter)
    .with_tools(Arc::new(tool_registry))
    .with_hooks(my_hook)
    .with_listener(my_listener);
```

`execute()` runs the full cycle: LLM call -> tool execution -> feed back -> repeat.

### Hooks

Intercept agent lifecycle events:

```rust
struct MyHook;

#[async_trait]
impl AgentHook for MyHook {
    async fn before_tool_call(&self, ctx: &HookContext, call: &ToolCall) -> HookResult {
        println!("Calling tool: {}", call.name);
        HookResult::Continue
    }
}
```

### Events

Receive streaming output via `EventListener`:

```rust
agent.with_on_event(|event| match event {
    AgentEvent::Delta(text) => print!("{}", text),
    AgentEvent::Thinking => println!("Thinking..."),
    AgentEvent::ThinkingDelta(reasoning) => print!("[reasoning] {}", reasoning),
    AgentEvent::ToolCallStart(call) => println!("Tool: {}", call.name),
    AgentEvent::Done => println!("Done."),
    _ => {}
});
```

### Multi-Agent Teams

Coordinate specialized agents:

```rust
let team = TeamAgent::builder()
    .provider(provider.clone())
    .add_agent(
        CollaborativeAgent::builder()
            .name("researcher")
            .role("Research specialist")
            .system_prompt("You research topics thoroughly.")
            .capability("search")
            .provider(provider.clone())
            .build(adapter.clone(), &bus)
    )
    .add_agent(
        CollaborativeAgent::builder()
            .name("writer")
            .role("Content writer")
            .system_prompt("You write clear, concise content.")
            .provider(provider)
            .build(adapter.clone(), &bus)
    )
    .build(adapter, &bus);

let result = team.execute_team_task("Write a brief on quantum computing").await;
```

### Sub-Agents

Register reusable agent components:

```rust
let mut registry = SubAgentRegistry::new();
registry.register(my_tool_agent);
let result = registry.execute("tool-agent", "do something", ctx).await?;
```

### Tool Registry

Register tools with JSON Schema definitions:

```rust
let mut tools = ToolRegistry::new();
tools.register(
    "search",
    "Search the web",
    json!({"type": "object", "properties": {"query": {"type": "string"}}}),
    |args, _pm| async move {
        let query = args["query"].as_str().unwrap_or("");
        Ok(format!("Results for: {}", query))
    },
);
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `extension` | no | SQLite storage, skills (YAML), RAG embeddings, session management |

```toml
[dependencies]
yoakore = { version = "0.1", features = ["extension"] }
```

## Module Structure

```
agent-foundation
├── agent          # BaseAgent, TeamAgent, SubAgent, AgentLike trait
├── hook           # AgentHook trait, HookContext, HookResult
├── provider       # OpenAIAdapter, embedding adapters, load balancing
├── runtime        # Skill selector (extension)
├── schema         # Config, Message, ToolCall, AgentEvent, Storage (extension)
├── tools          # ToolRegistry, ProcessManager
├── prelude        # One-import convenience module
├── error          # AgentError
├── llm            # LlmAdapter trait, AgentState
└── utils          # chunk_text, estimate_tokens
```
