use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Chat,
    Embedding,
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::Embedding => write!(f, "embedding"),
        }
    }
}

/// API 风格 — 决定 adapter 如何构建请求
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiStyle {
    #[default]
    Openai,
    Anthropic,
}

/// 思考/推理模式配置
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// 不显式配置，由 API 自行决定（对 o1/o3/Claude/DeepSeek 等模型默认开启思考）
    #[default]
    Default,
    /// 显式关闭思考（DeepSeek: {"thinking": {"type": "disabled"}}）
    Disabled,
    /// 自动 — 显式启用思考（OpenAI: reasoning_effort=medium, Anthropic: budget_tokens=10000）
    Auto,
    /// 基于努力程度（OpenAI/DeepSeek 风格: reasoning_effort）
    Effort { level: EffortLevel },
    /// 基于 token 预算（Anthropic/Gemini 风格）
    Budget { tokens: u32 },
}

/// 思考努力程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    /// DeepSeek 等支持的最高强度
    Max,
}

impl EffortLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    /// 将 effort 映射为近似 budget_tokens（用于跨 API 适配）
    pub fn to_budget_tokens(&self) -> u32 {
        match self {
            Self::Low => 2048,
            Self::Medium => 10000,
            Self::High => 32000,
            Self::Max => 64000,
        }
    }
}

impl ThinkingConfig {
    /// 获取 OpenAI 风格的 reasoning_effort 值
    pub fn to_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            Self::Default | Self::Disabled => None,
            Self::Auto => Some("medium"),
            Self::Effort { level } => Some(level.as_str()),
            Self::Budget { tokens } => {
                if *tokens <= 4096 {
                    Some("low")
                } else if *tokens <= 16384 {
                    Some("medium")
                } else {
                    Some("high")
                }
            }
        }
    }

    /// 获取 Anthropic 风格的 thinking 配置 JSON
    pub fn to_anthropic_thinking(&self) -> Option<serde_json::Value> {
        use serde_json::json;
        match self {
            Self::Default => None,
            Self::Disabled => Some(json!({"type": "disabled"})),
            Self::Auto => Some(json!({"type": "enabled", "budget_tokens": 10000})),
            Self::Effort { level } => {
                Some(json!({"type": "enabled", "budget_tokens": level.to_budget_tokens()}))
            }
            Self::Budget { tokens } => Some(json!({"type": "enabled", "budget_tokens": tokens})),
        }
    }

    /// 获取 OpenAI 风格的 thinking 开关（DeepSeek 等兼容 API 使用）
    ///
    /// 与 `to_reasoning_effort()` 配合：此方法控制开关，`to_reasoning_effort()` 控制强度。
    pub fn to_openai_thinking(&self) -> Option<serde_json::Value> {
        use serde_json::json;
        match self {
            Self::Default => None,
            Self::Disabled => Some(json!({"type": "disabled"})),
            Self::Auto | Self::Effort { .. } | Self::Budget { .. } => {
                Some(json!({"type": "enabled"}))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvider {
    /// 唯一 ID，用于区分不同 provider 实例
    pub id: String,
    // 模型类型（chat / embedding）
    pub kind: ModelKind,
    /// 供应商的名称，非常重要，会基于此字段进行负载均衡和调用策略
    pub name: String,
    /// API 基础 URL
    pub base_url: String,
    /// API Key 或 Token
    pub api_key: String,
    /// 模型名称或 ID，如 "gpt-4o-mini" 或 "text-embedding-3-small"
    pub model: String,
    /// 模型最大输出长度（token 数），用于调用前的适配和校验
    pub max_output: u32,
    /// 负载均衡权重，数值越大优先级越高
    pub weight: u32,
    /// 每分钟请求限制，用于调用节流
    pub requests_per_minute: u32,
    /// 模型能力等级：1=基础, 2=标准, 3=高级。用于分等级调用策略。
    #[serde(default = "default_tier")]
    pub tier: u8,
    /// 是否启用此 provider
    pub enabled: bool,
    /// API 风格（openai / anthropic）
    #[serde(default)]
    pub style: ApiStyle,
    /// 思考/推理模式配置
    #[serde(default)]
    pub thinking: ThinkingConfig,
    /// 模型上下文窗口大小（token 数），用于上下文管理
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,
}

impl ModelProvider {
    pub fn new(
        kind: ModelKind,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            kind,
            name: name.into(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            max_output: 4096,
            weight: 1,
            requests_per_minute: 60,
            tier: 1,
            enabled: true,
            style: ApiStyle::default(),
            thinking: ThinkingConfig::default(),
            max_context_tokens: 128000,
        }
    }
}

/// MCP 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

fn default_tier() -> u8 {
    1
}

fn default_max_context_tokens() -> u32 {
    128000
}

/// 一组同类型的 provider 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderGroup {
    /// 手动选中的 provider id (None = 负载均衡)。仅 embedding 有效。
    #[serde(default)]
    pub active: Option<String>,
    pub providers: Vec<ModelProvider>,
}

impl ProviderGroup {
    pub fn enabled(&self) -> Vec<&ModelProvider> {
        self.providers.iter().filter(|p| p.enabled).collect()
    }

    pub fn find(&self, id: &str) -> Option<&ModelProvider> {
        self.providers.iter().find(|p| p.id == id)
    }
}

/// Workspace 配置 — 仅指定路径，其余从工作空间目录读取
pub type WorkspaceConfig = Option<String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub chat: ProviderGroup,
    pub embedding: ProviderGroup,
    #[serde(default)]
    pub mcp: Vec<McpServerConfig>,
    /// 工作空间路径，None 时默认为 CWD/.ai-partner/
    #[serde(default)]
    pub workspace: WorkspaceConfig,
}

// Default 实现已移除 — 必须提供 config.json

impl AppConfig {
    pub fn config_path() -> PathBuf {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("config.json")
    }

    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path();
        if !path.exists() {
            return Err(ConfigError::NotFound(path.display().to_string()));
        }
        let content = std::fs::read_to_string(&path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    /// 按 kind 获取 provider group
    pub fn group(&self, kind: ModelKind) -> &ProviderGroup {
        match kind {
            ModelKind::Chat => &self.chat,
            ModelKind::Embedding => &self.embedding,
        }
    }

    /// 按 kind 获取可变 provider group
    pub fn group_mut(&mut self, kind: ModelKind) -> &mut ProviderGroup {
        match kind {
            ModelKind::Chat => &mut self.chat,
            ModelKind::Embedding => &mut self.embedding,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config file not found: {0}")]
    NotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AppConfig {
        AppConfig {
            chat: ProviderGroup {
                active: None,
                providers: vec![ModelProvider::new(
                    ModelKind::Chat,
                    "test",
                    "http://localhost",
                    "key",
                    "model",
                )],
            },
            embedding: ProviderGroup {
                active: None,
                providers: vec![ModelProvider::new(
                    ModelKind::Embedding,
                    "test",
                    "http://localhost",
                    "key",
                    "model",
                )],
            },
            mcp: Vec::new(),
            workspace: None,
        }
    }

    #[test]
    fn test_config_json_roundtrip() {
        let mut config = test_config();
        config.chat.providers.push(ModelProvider::new(
            ModelKind::Chat,
            "extra",
            "http://localhost",
            "key",
            "m",
        ));

        let json = serde_json::to_string_pretty(&config).unwrap();
        let loaded: AppConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.chat.providers.len(), 2);
        assert_eq!(loaded.embedding.providers.len(), 1);
    }

    #[test]
    fn test_group_by_kind() {
        let config = test_config();
        assert_eq!(config.group(ModelKind::Chat).providers.len(), 1);
        assert_eq!(config.group(ModelKind::Embedding).providers.len(), 1);
    }

    #[test]
    fn test_enabled_filter() {
        let mut config = test_config();
        config.chat.providers[0].enabled = true;
        let mut p2 = ModelProvider::new(ModelKind::Chat, "disabled", "http://x", "k", "m");
        p2.enabled = false;
        config.chat.providers.push(p2);

        assert_eq!(config.chat.enabled().len(), 1);
    }

    #[test]
    fn test_model_kind_json() {
        assert_eq!(serde_json::to_string(&ModelKind::Chat).unwrap(), "\"chat\"");
        assert_eq!(
            serde_json::to_string(&ModelKind::Embedding).unwrap(),
            "\"embedding\""
        );
    }

    #[test]
    fn test_load_real_config() {
        // Simulates deserializing the actual config.json format
        let json = r#"{
            "chat": {
                "providers": [
                    {
                        "id": "mimo-v2.5",
                        "kind": "chat",
                        "name": "xiaomi-tk",
                        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                        "api_key": "key",
                        "model": "mimo-v2.5",
                        "max_output": 1024000,
                        "weight": 5,
                        "requests_per_minute": 0,
                        "enabled": true
                    }
                ]
            },
            "embedding": { "active": "ollama-qwen3-embedding", "providers": [] }
        }"#;
        let config: AppConfig = serde_json::from_str(json).unwrap();
        assert!(config.chat.active.is_none());
        assert_eq!(config.chat.providers.len(), 1);
        assert_eq!(config.chat.providers[0].kind, ModelKind::Chat);
        assert_eq!(
            config.embedding.active.as_deref(),
            Some("ollama-qwen3-embedding")
        );
        assert!(config.embedding.providers.is_empty());
        assert!(config.workspace.is_none());
    }
}
