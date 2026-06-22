use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::schema::common::ModelProvider;
#[cfg(feature = "extension")]
use crate::schema::extension::storage::Storage;

/// Embedding adapter trait — abstracts different embedding backends.
#[async_trait]
pub trait EmbeddingAdapter: Send + Sync {
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts in one batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("No embedding provider configured")]
    NotConfigured,
}

// ── Ollama Embedding Adapter ──

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

/// Ollama embedding adapter — calls the local Ollama `/api/embeddings` endpoint.
pub struct OllamaEmbeddingAdapter {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaEmbeddingAdapter {
    pub fn new(provider: &ModelProvider) -> Self {
        Self {
            client: Client::new(),
            base_url: provider.base_url.trim_end_matches('/').to_string(),
            model: provider.model.clone(),
        }
    }
}

#[async_trait]
impl EmbeddingAdapter for OllamaEmbeddingAdapter {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = OllamaEmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let resp = self.client.post(&url).json(&body).send().await?;
        let embed_resp: OllamaEmbedResponse = resp.json().await?;
        Ok(embed_resp.embedding)
    }
}

// ── OpenAI-compatible Embedding Adapter ──

#[derive(Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedData>,
}

#[derive(Deserialize)]
struct OpenAIEmbedData {
    embedding: Vec<f32>,
}

/// OpenAI-compatible embedding adapter — works with any OpenAI-compatible API.
pub struct OpenAIEmbeddingAdapter {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAIEmbeddingAdapter {
    pub fn new(provider: &ModelProvider) -> Self {
        Self {
            client: Client::new(),
            base_url: provider.base_url.trim_end_matches('/').to_string(),
            api_key: provider.api_key.clone(),
            model: provider.model.clone(),
        }
    }
}

#[async_trait]
impl EmbeddingAdapter for OpenAIEmbeddingAdapter {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = OpenAIEmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let embed_resp: OpenAIEmbedResponse = resp.json().await?;
        embed_resp
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| EmbeddingError::Provider("empty response".into()))
    }
}

// ── Factory ──

/// Create an embedding adapter from a provider configuration.
pub fn create_embedding_adapter(provider: &ModelProvider) -> Arc<dyn EmbeddingAdapter> {
    // Ollama typically uses localhost:11434 and no API key
    if provider.base_url.contains("11434") || provider.api_key.is_empty() {
        Arc::new(OllamaEmbeddingAdapter::new(provider))
    } else {
        Arc::new(OpenAIEmbeddingAdapter::new(provider))
    }
}

// ── Embedding Pipeline ──

/// Embed all document chunks for a conversation that are missing embeddings.
#[cfg(feature = "extension")]
pub async fn embed_missing_documents(
    storage: &Arc<Storage>,
    adapter: &dyn EmbeddingAdapter,
    session_id: &str,
) -> Result<usize, EmbeddingError> {
    let docs = storage
        .get_documents_by_session(session_id)
        .map_err(|e| EmbeddingError::Provider(e.to_string()))?;

    let mut count = 0;
    for doc in &docs {
        if doc.embedding.is_some() {
            continue;
        }
        let embedding = adapter.embed(&doc.content).await?;
        storage
            .save_document_embedding(&doc.id, &embedding)
            .map_err(|e| EmbeddingError::Provider(e.to_string()))?;
        count += 1;
    }
    Ok(count)
}
