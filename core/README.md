# yoakore

A modular, extensible foundation for building AI agents in Rust.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Your Application                        │
├──────────────────────────────────────────────────────────────┤
│  AgentBuilder (unified constructor)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │BaseAgent │  │TeamAgent │  │PlanAgent │                   │
│  │(single)  │  │(multi-   │  │(task     │                   │
│  │          │  │ agent)   │  │decompose)│                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
├──────────────────────────────────────────────────────────────┤
│  AgentLike │ AgentHook │ EventListener │ Session / Conversation│
├──────────────────────────────────────────────────────────────┤
│  LlmAdapter │ RateLimitedAdapter │ ProviderBalancer │ Semaphore│
├──────────────────────────────────────────────────────────────┤
│  ToolRegistry │ ProcessManager │ ApprovalPolicy │ SubAgentRegistry│
├──────────────────────────────────────────────────────────────┤
│  Config │ Message │ ToolCall │ AgentEvent │ State             │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use yoakore::prelude::*;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    // 1. Configure a provider
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "my-llm",
        "https://api.openai.com/v1",
        "sk-your-api-key",
        "gpt-4o-mini",
    );

    // 2. Build agent with AgentBuilder
    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .max_rounds(10)
        .with_on_event(|event| {
            if let AgentEvent::Delta(text) = event {
                print!("{}", text);
            }
        })
        .build_base();

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
{ "type": "default" }                           // not explicitly configured, API decides
{ "type": "disabled" }                          // explicitly disable thinking
{ "type": "auto" }                              // enable, medium strength
{ "type": "effort", "level": "low" }            // low / medium / high / max
{ "type": "effort", "level": "max" }            // maximum intensity (DeepSeek etc.)
{ "type": "budget", "tokens": 10000 }           // token budget (Anthropic style)
```

Cross-API mapping is automatic — effort levels map to approximate budgets and vice versa.

For DeepSeek, thinking is enabled by default. Use `disabled` to turn it off, or `effort` with `max` for maximum reasoning depth.

## Core Concepts

### Session & Conversation

Manage multi-turn conversations with `Session`:

```rust
let mut session = Session::with_system("You are a helpful assistant.");
session.add_user("What is Rust?");
session.add_assistant("Rust is a systems programming language...");

// Session manages history, system prompt, and metadata
let messages = session.messages_with_system(); // includes system prompt
session.truncate(50); // keep last 50 messages for context window
```

Use `execute_in_session()` for seamless multi-turn with BaseAgent:

```rust
let mut session = Session::with_system("You are a helpful assistant.");
session.add_user("Hello!");
let reply = agent.execute_in_session(&provider, &mut session).await?;
session.add_user("Tell me more.");
let reply = agent.execute_in_session(&provider, &mut session).await?;
```

### BaseAgent

The foundation agent with a built-in tool-use loop. Build with `AgentBuilder`:

```rust
let agent = AgentBuilder::new()
    .provider(provider.clone())
    .max_rounds(10)
    .tools(Arc::new(tool_registry))
    .hooks(my_hook)
    .listener(my_listener)
    .build_base();
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

`HookResult` variants: `Continue` (proceed), `Skip` (skip this operation), `Denied(msg)` (soft deny — returns tool result but continues loop), `Abort(msg)` (stop the agent loop).

Compose multiple hooks with `ComposedHook`:

```rust
let composed = ComposedHook { first: hook_a, second: hook_b };
// first runs first; if it returns Continue, second runs
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

Coordinate specialized agents that communicate through `ask_peer` and `report_result`:

```rust
let adapter = Arc::new(OpenAIAdapter::new());
let team = AgentBuilder::new()
    .provider(provider.clone())
    .build_team(vec![
        AgentBuilder::team_agent(&adapter, "researcher", "Research specialist",
            "You research topics thoroughly."),
        AgentBuilder::team_agent(&adapter, "writer", "Content writer",
            "You write clear, concise content."),
    ]);

let result = team.execute_team_task("Write a brief on quantum computing").await;
println!("{}", result.synthesis);
```

Agents can ask each other questions via `ask_peer`. The coordinator routes requests and feeds answers back. Cycle detection prevents infinite recursion.

### PlanAgent

Decompose complex tasks into a plan and delegate to sub-agents:

```rust
let mut registry = SubAgentRegistry::new();
registry.register(my_search_agent);
registry.register(my_writer_agent);

let plan_agent = AgentBuilder::new()
    .provider(provider.clone())
    .build_plan();

let result = plan_agent
    .execute_plan("Research and summarize quantum computing", &registry)
    .await?;

println!("{}", result.synthesis);
// result.plan — the generated subtasks
// result.subtask_results — per-subtask outputs
```

PlanAgent uses the LLM to:
1. Generate a JSON plan with subtasks and dependency relationships
2. Execute subtasks respecting dependency order (independent tasks run concurrently)
3. Synthesize all results into a coherent response

PlanAgent implements `SubAgent`, so it can be nested inside other PlanAgents.

### AgentBuilder

Unified builder for constructing any agent type. The recommended entry point for all agent construction.

**Single provider:**

```rust
let agent = AgentBuilder::new()
    .provider(provider)
    .max_rounds(20)
    .tools(tools)
    .build_base();
```

**Multiple providers with load balancing:**

```rust
let (agent, balancer) = AgentBuilder::new()
    .providers(vec![provider_a, provider_b, provider_c])
    .max_rounds(20)
    .build_base_balanced();

// Each call auto-selects a provider
let reply = agent.execute_balanced(&balancer, &mut messages).await?;
```

**Build any agent type from the same configuration:**

```rust
let builder = AgentBuilder::new()
    .providers(vec![provider_a, provider_b])
    .max_rounds(20);

let base = builder.build_base();          // single agent
let team = builder.build_team(agents);    // multi-agent team
let plan = builder.build_plan();          // plan/decompose agent
```

All adapters are automatically wrapped with `RateLimitedAdapter` to enforce per-provider rate limiting based on `requests_per_minute`.

### Tool Approval Policy

By default, all tools execute without approval. Set an `ApprovalPolicy` on `ToolRegistry` to require user confirmation for specific tools:

```rust
use std::sync::Arc;
use yoakore::prelude::*;

// Define an async approval callback
let callback: ApprovalCallback = Arc::new(|name, args| {
    Box::pin(async move {
        // In a real app, show a UI dialog and return the user's decision.
        println!("Approve tool '{name}'? (args: {args})");
        true
    })
});

// Create tools with approval policy
let mut tools = ToolRegistry::new();
tools.register(my_tool_def, my_handler);
tools.set_approval(ApprovalPolicy::require_approval(
    ["shell_execute", "write_file"],
    callback,
));

// Pass tools to agent — approval is checked automatically
let agent = AgentBuilder::new()
    .provider(provider)
    .tools(Arc::new(tools))
    .build_base();
```

**Approval scopes:**

| Scope | Behavior |
|-------|----------|
| `PerCall` | Ask the user every time the tool is called |
| `Session` | Ask once; cache the approval for the agent's lifetime |

Use the builder for mixed scopes:

```rust
let policy = ApprovalPolicy::builder(callback)
    .require("shell_execute")           // PerCall: ask every time
    .require_session("web_search")      // Session: ask once
    .build();
tools.set_approval(policy);
```

Denied tools produce a `"[denied]"` tool result and the agent loop continues (soft deny). Tools not listed in any rule are auto-approved — the default behavior is unchanged.

### Rate Limiting & Multi-Provider Load Balancing

All adapters built through `AgentBuilder` are automatically wrapped with `RateLimitedAdapter`, which enforces per-provider rate limiting using a token-bucket `Semaphore` configured from `requests_per_minute`.

For multi-provider setups, `ProviderBalancer` automatically selects providers using:
- **Round-robin** — when all providers have equal weight
- **Weighted round-robin** — when weights differ (higher weight = more traffic)

Disabled providers (`enabled: false`) are automatically skipped.

```rust
let mut p1 = ModelProvider::new(ModelKind::Chat, "openai", "https://api.openai.com/v1", "key1", "gpt-4o");
p1.weight = 5;
p1.requests_per_minute = 60;

let mut p2 = ModelProvider::new(ModelKind::Chat, "deepseek", "https://api.deepseek.com/v1", "key2", "deepseek-chat");
p2.weight = 1;
p2.requests_per_minute = 30;

// Weighted: openai gets ~5x more traffic than deepseek
// Each provider independently rate-limited at its own RPM
let (agent, balancer) = AgentBuilder::new()
    .providers(vec![p1, p2])
    .build_base_balanced();
```

### Sub-Agents

Register reusable agent components with capabilities:

```rust
let mut registry = SubAgentRegistry::new();
registry.register(my_tool_agent);

// Execute by name
let result = registry.execute("tool-agent", "do something", ctx).await?;

// Batch execution
let results = registry.execute_sequential(
    &[("agent-a", "task 1"), ("agent-b", "task 2")],
    session_id,
    &config,
).await;

let results = registry.execute_parallel(
    &[("agent-a", "task 1"), ("agent-b", "task 2")],
    session_id,
    &config,
).await;

// Introspect capabilities
let caps = registry.list_capabilities(); // HashMap<String, Vec<String>>
```

### Tool Registry

Register tools with JSON Schema definitions. Optionally set an `ApprovalPolicy` to require user confirmation for specific tools (see [Tool Approval Policy](#tool-approval-policy)):

```rust
let mut tools = ToolRegistry::new();
tools.register(
    ToolDefinition {
        name: "search".into(),
        description: "Search the web".into(),
        parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
    },
    |args| {
        let query = args["query"].as_str().unwrap_or("");
        Ok(format!("Results for: {}", query))
    },
);

// Optional: require approval for sensitive tools
tools.set_approval(ApprovalPolicy::require_approval(["delete_file"], callback));
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `skills` | no | Skill loading from YAML (SKILL.md) |
| `storage` | no | SQLite-backed session/conversation/document persistence |
| `extension` | no | Meta-feature: enables both `skills` and `storage` |

```toml
# Enable everything
[dependencies]
yoakore = { version = "0.2", features = ["extension"] }

# Or pick only what you need
[dependencies]
yoakore = { version = "0.2", features = ["skills"] }
```

## Module Structure

```
yoakore
├── agent
│   ├── base         # BaseAgent — single agent with tool loop
│   ├── team         # TeamAgent, CollaborativeAgent — multi-agent coordination
│   ├── plan         # PlanAgent — task decomposition and delegation
│   ├── subagent     # SubAgent trait, SubAgentRegistry, batch execution
│   ├── builder      # AgentBuilder — unified constructor
│   └── mod.rs       # AgentLike, ToolExecutor traits
├── hook             # AgentHook, HookResult (Continue/Skip/Denied/Abort), ComposedHook
├── provider         # OpenAIAdapter, embedding adapters, load balancing
├── runtime          # Skill selector (extension)
├── schema           # Config, Message, ToolCall, AgentEvent, Storage (extension)
├── tools
│   ├── registry     # ToolRegistry — tool registration and execution
│   ├── process      # ProcessManager — subprocess management
│   └── policy       # ApprovalPolicy, ApprovalCallback — tool approval rules
├── prelude          # One-import convenience module
├── error            # AgentError
├── llm              # LlmAdapter trait, AgentState, Session, Conversation
└── utils            # chunk_text, estimate_tokens
```

## Examples

| Example | Description |
|---------|-------------|
| `single_turn_cli` | One-shot agent with calculator and time tools |
| `multi_turn_cli` | Multi-turn REPL with Session management |
| `message_injection` | Inject user messages mid-run via channel |
| `sub_agent` | SubAgent registry and execution |
| `team_agent` | Multi-agent team coordination |
