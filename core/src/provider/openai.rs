use async_trait::async_trait;
use bytes::Bytes;
use futures_core::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::error::AgentError;
use crate::llm::adapter::{AgentResponse, LlmAdapter, Usage};
use crate::schema::common::EventListener;
use crate::schema::common::{AgentEvent, Message, ModelProvider, Role, ToolCall, ToolDefinition};

pub struct OpenAIAdapter {
    client: Client,
}

impl OpenAIAdapter {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl Default for OpenAIAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request types ──

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatTool>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    /// OpenAI style: reasoning_effort ("low" / "medium" / "high")
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    /// Anthropic style: thinking config
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct ChatTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatFunction,
}

#[derive(Serialize)]
struct ChatFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ChatFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatFunctionCall {
    name: String,
    arguments: String,
}

// ── Streaming response types ──

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Deserialize)]
struct StreamUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    index: usize,
    id: Option<String>,
    function: Option<StreamFunctionCall>,
}

#[derive(Deserialize)]
struct StreamFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

fn build_messages(messages: &[Message]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            ChatMessage {
                role: role.to_string(),
                content: Some(m.content.clone()),
                tool_calls: m.tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .map(|tc| ChatToolCall {
                            id: tc.id.clone(),
                            call_type: "function".into(),
                            function: ChatFunctionCall {
                                name: tc.name.clone(),
                                arguments: tc.arguments.to_string(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<ChatTool> {
    tools
        .iter()
        .map(|t| ChatTool {
            tool_type: "function".into(),
            function: ChatFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

/// 累积流式工具调用的中间状态
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

#[async_trait]
impl LlmAdapter for OpenAIAdapter {
    async fn chat(
        &self,
        provider: &ModelProvider,
        messages: &[Message],
        tools: &[ToolDefinition],
        listener: &dyn EventListener,
    ) -> Result<AgentResponse, AgentError> {
        let url = format!(
            "{}/chat/completions",
            provider.base_url.trim_end_matches('/')
        );

        let chat_messages = build_messages(messages);
        let chat_tools = if tools.is_empty() {
            None
        } else {
            Some(build_tools(tools))
        };

        // Build thinking config based on style
        // OpenAI style: thinking (toggle) + reasoning_effort (intensity)
        // Anthropic style: thinking (combined toggle + budget)
        let (reasoning_effort, thinking) = match provider.style {
            crate::schema::common::ApiStyle::Openai => (
                provider.thinking.to_reasoning_effort().map(String::from),
                provider.thinking.to_openai_thinking(),
            ),
            crate::schema::common::ApiStyle::Anthropic => {
                (None, provider.thinking.to_anthropic_thinking())
            }
        };

        let request = ChatRequest {
            model: provider.model.clone(),
            messages: chat_messages,
            max_tokens: Some(provider.max_output),
            tools: chat_tools,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            reasoning_effort,
            thinking,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::Other(format!(
                "OpenAI API error {status}: {body}"
            )));
        }

        // 流式读取 SSE
        let stream = response.bytes_stream();
        let reader = StreamReader::new(stream);
        let buf_reader = BufReader::new(reader);

        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut tool_accumulators: Vec<ToolCallAccumulator> = Vec::new();
        let mut has_tool_calls = false;
        let mut last_usage: Option<StreamUsage> = None;

        let mut lines = buf_reader.lines();
        while let Some(line) = lines.next_line().await? {
            let line = line.trim();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if line == "data: [DONE]" {
                break;
            }

            let data = match line.strip_prefix("data: ") {
                Some(d) => d,
                None => continue,
            };

            let chunk: StreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // 提取 usage（通常在最后一个 chunk）
            if chunk.usage.is_some() {
                last_usage = chunk.usage;
            }

            for choice in chunk.choices {
                // 内容 delta
                if let Some(content) = choice.delta.content {
                    full_content.push_str(&content);
                    listener.on_event(&AgentEvent::Delta(content));
                }

                // 推理内容 delta
                if let Some(reasoning) = choice.delta.reasoning_content {
                    full_reasoning.push_str(&reasoning);
                    listener.on_event(&AgentEvent::ThinkingDelta(reasoning));
                }

                // 工具调用 delta
                if let Some(tc_deltas) = choice.delta.tool_calls {
                    has_tool_calls = true;
                    for delta in tc_deltas {
                        while tool_accumulators.len() <= delta.index {
                            tool_accumulators.push(ToolCallAccumulator {
                                id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            });
                        }
                        let acc = &mut tool_accumulators[delta.index];
                        if let Some(id) = delta.id {
                            acc.id = id;
                        }
                        if let Some(func) = delta.function {
                            if let Some(name) = func.name {
                                acc.name = name;
                            }
                            if let Some(args) = func.arguments {
                                acc.arguments.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        // 将 StreamUsage 转为 Usage
        let usage = last_usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens.unwrap_or(0),
            completion_tokens: u.completion_tokens.unwrap_or(0),
            total_tokens: u.total_tokens.unwrap_or(0),
            cached_input_tokens: u
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
        });

        // 构建最终响应
        if has_tool_calls && !tool_accumulators.is_empty() {
            let calls: Vec<ToolCall> = tool_accumulators
                .into_iter()
                .map(|acc| {
                    let args: Value = serde_json::from_str(&acc.arguments)
                        .unwrap_or(Value::Object(serde_json::Map::new()));
                    ToolCall {
                        id: acc.id,
                        name: acc.name,
                        arguments: args,
                    }
                })
                .collect();

            let mut assistant_msg = Message::assistant("");
            assistant_msg.reasoning_content = if full_reasoning.is_empty() {
                None
            } else {
                Some(full_reasoning)
            };
            assistant_msg.tool_calls = Some(calls.clone());
            Ok(AgentResponse::tool_calls(calls).with_usage(usage.unwrap_or_default()))
        } else {
            let mut assistant_msg = Message::assistant(&full_content);
            assistant_msg.reasoning_content = if full_reasoning.is_empty() {
                None
            } else {
                Some(full_reasoning)
            };
            Ok(
                AgentResponse::message_complete(assistant_msg)
                    .with_usage(usage.unwrap_or_default()),
            )
        }
    }
}

/// 将 reqwest 的 bytes_stream 包装为 AsyncRead
struct StreamReader {
    inner: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: Vec<u8>,
    pos: usize,
}

impl StreamReader {
    fn new(stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
            buffer: Vec::new(),
            pos: 0,
        }
    }
}

impl tokio::io::AsyncRead for StreamReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        if self.pos < self.buffer.len() {
            let remaining = &self.buffer[self.pos..];
            let to_copy = remaining.len().min(buf.remaining());
            buf.put_slice(&remaining[..to_copy]);
            self.pos += to_copy;
            if self.pos >= self.buffer.len() {
                self.buffer.clear();
                self.pos = 0;
            }
            return std::task::Poll::Ready(Ok(()));
        }

        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                self.buffer = bytes.to_vec();
                self.pos = 0;
                let to_copy = self.buffer.len().min(buf.remaining());
                buf.put_slice(&self.buffer[..to_copy]);
                self.pos = to_copy;
                if self.pos >= self.buffer.len() {
                    self.buffer.clear();
                    self.pos = 0;
                }
                std::task::Poll::Ready(Ok(()))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                std::task::Poll::Ready(Err(std::io::Error::other(e)))
            }
            std::task::Poll::Ready(None) => std::task::Poll::Ready(Ok(())),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
