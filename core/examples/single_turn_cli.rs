use std::io::{self, Write};
use std::sync::Arc;

use yoakore::prelude::*;
use serde_json::json;

/// 简易单轮 CLI Agent 演示
///
/// 用法:
///   cargo run -p agent-core --example single_turn_cli
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. 配置 ModelProvider ──
    let provider = ModelProvider::new(
        ModelKind::Chat,
        "cli-demo",
        "https://api.openai.com/v1", // 替换为你的 API 地址
        "sk-your-api-key-here",      // 替换为你的 API Key
        "gpt-4o-mini",               // 替换为你的模型名称
    );

    // ── 2. 注册示例工具 ──
    let mut tools = ToolRegistry::new();

    // 计算器工具
    tools.register(
        ToolDefinition {
            name: "calculator".into(),
            description: "计算数学表达式，返回数值结果。支持加减乘除和括号。".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 (1 + 2) * 3"
                    }
                },
                "required": ["expression"]
            }),
        },
        |args| {
            let expr = args["expression"].as_str().ok_or("缺少 expression 参数")?;
            let result = eval_simple_math(expr).map_err(|e| e.to_string())?;
            Ok(format!("{result}"))
        },
    );

    // 当前时间工具
    tools.register(
        ToolDefinition {
            name: "current_time".into(),
            description: "获取当前日期和时间".into(),
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

    // ── 3. 读取用户输入 ──
    print!("你: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_string();
    if input.is_empty() {
        println!("未输入内容，退出。");
        return Ok(());
    }

    // ── 4. 构建 Agent 并执行 ──
    let adapter = OpenAIAdapter::new();
    let agent = BaseAgent::new(adapter)
        .with_max_rounds(10)
        .with_tools(Arc::new(tools))
        .with_on_event(|event| match event {
            AgentEvent::Delta(text) => print!("{text}"),
            AgentEvent::ThinkingDelta(text) => eprint!("{text}"),
            AgentEvent::ToolCallStart(tc) => {
                eprint!("\n[调用工具: {}]", tc.name);
            }
            AgentEvent::ToolCallResult { result, .. } => {
                eprintln!(" -> {result}");
            }
            AgentEvent::Done => {
                eprintln!();
            }
            _ => {}
        });

    let mut messages = vec![Message::user(&input)];

    println!("助手: ");
    let reply = agent.execute(&provider, &mut messages).await?;

    // 如果流式输出未生效（非流式 fallback），直接打印结果
    if reply.is_empty() {
        println!("(无回复)");
    }

    Ok(())
}

/// 简易四则运算求值器（仅支持 +, -, *, / 和括号）
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
                let n: f64 = s.parse().map_err(|_| format!("无效数字: {s}"))?;
                tokens.push(Token::Num(n));
                continue;
            }
            other => return Err(format!("未知字符: {other}")),
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
                    return Err("除零错误".into());
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
        return Err("表达式不完整".into());
    }
    match &tokens[pos] {
        Token::Num(n) => Ok((*n, pos + 1)),
        Token::LParen => {
            let (val, next) = parse_expr(tokens, pos + 1)?;
            if next >= tokens.len() || !matches!(&tokens[next], Token::RParen) {
                return Err("缺少右括号".into());
            }
            Ok((val, next + 1))
        }
        Token::Op('-') => {
            let (val, next) = parse_factor(tokens, pos + 1)?;
            Ok((-val, next))
        }
        other => Err(format!("意外的 token: {other:?}")),
    }
}
