use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::tool::ToolCall;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    /// assistant 消息的推理内容
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// assistant 消息携带的工具调用请求
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// tool role 消息关联的 call id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role,
            content: content.into(),
            timestamp: Utc::now(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// assistant 消息，携带工具调用请求（content 可能为空）
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: Role::Assistant,
            content: String::new(),
            timestamp: Utc::now(),
            reasoning_content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// tool role 消息，携带工具执行结果
    pub fn tool_result(call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            role: Role::Tool,
            content: result.into(),
            timestamp: Utc::now(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
        }
    }
}
