#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("rate limited, retry after {retry_after_secs:.1}s")]
    RateLimited { retry_after_secs: f64 },

    #[error("{0}")]
    Other(String),
}
