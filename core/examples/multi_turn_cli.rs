use std::io::{self, Write};
use std::sync::Arc;

use serde_json::json;
use yoakore::prelude::*;

/// Multi-turn CLI Agent demo using Session for conversation management.
///
/// Usage:
///   cargo run -p yoakore --example multi_turn_cli
///
/// Commands:
///   /quit    — exit
///   /clear   — clear conversation history
///   /history — show message count
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Configure ModelProvider ──
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "cli-demo",
        "https://api.openai.com/v1", // replace with your API endpoint
        "sk-your-api-key-here",      // replace with your API key
        "gpt-4o-mini",               // replace with your model name
    );

    // ── 2. Register tools ──
    let mut tools = ToolRegistry::new();

    tools.register(
        ToolDefinition {
            name: "calculator".into(),
            description: "Evaluate a math expression and return the numeric result.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. (1 + 2) * 3"
                    }
                },
                "required": ["expression"]
            }),
        },
        |args| {
            let expr = args["expression"].as_str().ok_or("missing expression")?;
            let result = eval_simple_math(expr).map_err(|e| e.to_string())?;
            Ok(format!("{result}"))
        },
    );

    tools.register(
        ToolDefinition {
            name: "current_time".into(),
            description: "Get the current date and time.".into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        },
        |_args| {
            let now = chrono::Local::now();
            Ok(now.format("%Y-%m-%d %H:%M:%S").to_string())
        },
    );

    let tools = Arc::new(tools);

    // ── 3. Create Session and Agent ──
    let mut session = Session::with_system(
        "You are a helpful assistant with access to a calculator and clock. \
         Keep responses concise.",
    );

    let agent = AgentBuilder::new()
        .provider(provider.clone())
        .max_rounds(10)
        .tools(tools)
        .with_on_event(|event| match event {
            AgentEvent::Delta(text) => print!("{text}"),
            AgentEvent::ThinkingDelta(text) => eprint!("{text}"),
            AgentEvent::ToolCallStart(tc) => {
                eprint!("\n[tool: {}]", tc.name);
            }
            AgentEvent::ToolCallResult { result, .. } => {
                eprintln!(" -> {result}");
            }
            AgentEvent::Done => {
                eprintln!();
            }
            _ => {}
        })
        .build_base();

    // ── 4. REPL ──
    println!("Multi-turn CLI Agent (type /quit to exit, /clear to reset, /history for stats)");
    println!();

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        // Handle commands
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
            "/history" => {
                println!(
                    "(messages: {}, session: {})\n",
                    session.len(),
                    &session.id[..8]
                );
                continue;
            }
            _ => {}
        }

        // Add user message and run agent
        session.add_user(&input);

        println!("Assistant: ");
        let _reply = agent.execute_in_session(&provider, &mut session).await?;
        println!();
    }

    Ok(())
}

/// Simple math expression evaluator (supports +, -, *, / and parentheses)
fn eval_simple_math(expr: &str) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    let (result, _) = parse_expr(&tokens, 0)?;
    Ok(result)
}

#[derive(Debug)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(expr: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' => {}
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            op @ ('+' | '-' | '*' | '/') => tokens.push(Token::Op(op)),
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let n: f64 = s.parse().map_err(|_| format!("invalid number: {s}"))?;
                tokens.push(Token::Num(n));
                continue;
            }
            other => return Err(format!("unexpected character: {other}")),
        }
        i += 1;
    }
    Ok(tokens)
}

fn parse_expr(tokens: &[Token], pos: usize) -> Result<(f64, usize), String> {
    let (mut lhs, mut pos) = parse_term(tokens, pos)?;
    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Op('+') => {
                let (rhs, next) = parse_term(tokens, pos + 1)?;
                lhs += rhs;
                pos = next;
            }
            Token::Op('-') => {
                let (rhs, next) = parse_term(tokens, pos + 1)?;
                lhs -= rhs;
                pos = next;
            }
            _ => break,
        }
    }
    Ok((lhs, pos))
}

fn parse_term(tokens: &[Token], pos: usize) -> Result<(f64, usize), String> {
    let (mut lhs, mut pos) = parse_factor(tokens, pos)?;
    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Op('*') => {
                let (rhs, next) = parse_factor(tokens, pos + 1)?;
                lhs *= rhs;
                pos = next;
            }
            Token::Op('/') => {
                let (rhs, next) = parse_factor(tokens, pos + 1)?;
                if rhs == 0.0 {
                    return Err("division by zero".into());
                }
                lhs /= rhs;
                pos = next;
            }
            _ => break,
        }
    }
    Ok((lhs, pos))
}

fn parse_factor(tokens: &[Token], pos: usize) -> Result<(f64, usize), String> {
    if pos >= tokens.len() {
        return Err("incomplete expression".into());
    }
    match &tokens[pos] {
        Token::Num(n) => Ok((*n, pos + 1)),
        Token::LParen => {
            let (val, next) = parse_expr(tokens, pos + 1)?;
            if next >= tokens.len() || !matches!(&tokens[next], Token::RParen) {
                return Err("missing closing parenthesis".into());
            }
            Ok((val, next + 1))
        }
        Token::Op('-') => {
            let (val, next) = parse_factor(tokens, pos + 1)?;
            Ok((-val, next))
        }
        other => Err(format!("unexpected token: {other:?}")),
    }
}
