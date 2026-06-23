use std::sync::Arc;

use async_trait::async_trait;

use yoakore::prelude::*;

/// A memory entry retrieved from the memory store.
///
/// This is the craft-level abstraction — storage backends map their own
/// representations into this struct.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub title: String,
    pub content: String,
    pub tags: Option<String>,
    pub weight: f64,
}

/// A new memory extracted from an assistant reply.
pub struct NewMemory {
    pub title: String,
    pub content: String,
    pub tags: Option<String>,
}

/// Memory provider trait — defines how memories are retrieved, extracted, and saved.
///
/// Implement this trait to fully replace the memory system behavior.
pub trait MemoryProvider: Send + Sync {
    /// Retrieve memories relevant to the current conversation.
    fn retrieve<'a>(
        &'a self,
        query: &'a str,
        limit: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<MemoryEntry>> + Send + 'a>>;

    /// Extract a new memory worth saving from an assistant reply.
    /// Returns `None` if there is nothing worth remembering.
    fn extract<'a>(
        &'a self,
        assistant_reply: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<NewMemory>> + Send + 'a>>;

    /// Persist an extracted memory. Called after `extract` returns `Some`.
    fn save<'a>(
        &'a self,
        memory: &'a NewMemory,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>>;
}

/// Memory system configuration.
pub struct MemoryConfig {
    /// Maximum number of relevant memories to inject per LLM call.
    pub max_injected: usize,
    /// Whether to automatically extract memories from assistant replies.
    pub auto_extract: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_injected: 5,
            auto_extract: true,
        }
    }
}

// ── SQLite-backed MemoryStore (optional) ──

#[cfg(feature = "storage")]
mod storage_impl {
    use super::*;
    use rusqlite::{Connection, params};
    use std::path::PathBuf;
    use std::sync::Mutex;

    const DECAY_RATE: f64 = 0.05;

    /// [`MemoryProvider`] implementation with its own SQLite connection.
    ///
    /// Manages a `memories` table independently — no dependency on core's `Storage`.
    /// Supports weighted memories with exponential forgetting curve decay.
    pub struct MemoryStore {
        conn: Mutex<Connection>,
        auto_extract: bool,
    }

    impl MemoryStore {
        /// Open or create a memory database at the given path.
        pub fn new(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let conn = Connection::open(path)?;
            let store = Self {
                conn: Mutex::new(conn),
                auto_extract: true,
            };
            store.init_tables()?;
            Ok(store)
        }

        /// Open or create at the default data directory.
        pub fn new_default() -> Result<Self, Box<dyn std::error::Error>> {
            let base = dirs::data_dir()
                .or_else(dirs::config_dir)
                .or_else(dirs::home_dir)
                .unwrap_or_else(|| PathBuf::from("."));
            Self::new(&base.join("ai-partner").join("conversations.db"))
        }

        pub fn auto_extract(mut self, enabled: bool) -> Self {
            self.auto_extract = enabled;
            self
        }

        fn init_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
            self.conn.lock().unwrap().execute_batch(
                "CREATE TABLE IF NOT EXISTS memories (
                    id                TEXT PRIMARY KEY,
                    title             TEXT NOT NULL,
                    content           TEXT NOT NULL,
                    tags              TEXT,
                    weight            REAL NOT NULL DEFAULT 1.0,
                    last_activated_at TEXT NOT NULL,
                    activation_count  INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_title ON memories(title);
                CREATE INDEX IF NOT EXISTS idx_memories_weight ON memories(weight DESC);",
            )?;
            Ok(())
        }

        fn calc_decay(weight: f64, last_activated_at: &str) -> f64 {
            let now = chrono::Utc::now();
            if let Ok(last) = chrono::DateTime::parse_from_rfc3339(last_activated_at) {
                let days = (now - last.with_timezone(&chrono::Utc)).num_days().max(0) as f64;
                weight * (1.0 - DECAY_RATE).powf(days)
            } else {
                weight
            }
        }

        /// Search memories by keyword with real-time weight decay.
        pub fn search(&self, query: &str, limit: usize) -> Vec<MemoryEntry> {
            let pattern = format!("%{query}%");
            let conn = self.conn.lock().unwrap();
            let mut stmt = match conn.prepare(
                "SELECT id, title, content, tags, weight, last_activated_at
                 FROM memories WHERE title LIKE ?1 OR content LIKE ?1 OR tags LIKE ?1
                 ORDER BY weight DESC LIMIT ?2",
            ) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let rows = match stmt.query_map(params![pattern, limit as i64], |row| {
                let id: String = row.get(0)?;
                let title: String = row.get(1)?;
                let content: String = row.get(2)?;
                let tags: Option<String> = row.get(3)?;
                let weight: f64 = row.get(4)?;
                let last_activated: String = row.get(5)?;
                Ok((id, title, content, tags, weight, last_activated))
            }) {
                Ok(r) => r,
                Err(_) => return Vec::new(),
            };

            let mut entries = Vec::new();
            for row in rows.flatten() {
                let decayed = Self::calc_decay(row.4, &row.5);
                let _ = conn.execute(
                    "UPDATE memories SET weight = ?2 WHERE id = ?1",
                    params![row.0, decayed],
                );
                entries.push(MemoryEntry {
                    title: row.1,
                    content: row.2,
                    tags: row.3,
                    weight: decayed,
                });
            }
            entries.sort_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            entries
        }

        /// Save a new memory entry.
        pub fn save_entry(&self, title: &str, content: &str, tags: Option<&str>) {
            let now = chrono::Utc::now().to_rfc3339();
            let id = uuid::Uuid::new_v4().to_string();
            if let Err(e) = self.conn.lock().unwrap().execute(
                "INSERT INTO memories (id, title, content, tags, weight, last_activated_at, activation_count, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 1.0, ?5, 0, ?5, ?5)",
                params![id, title, content, tags, now],
            ) {
                log::warn!("MemoryStore: failed to save memory: {}", e);
            }
        }
    }

    impl MemoryProvider for MemoryStore {
        fn retrieve<'a>(
            &'a self,
            query: &'a str,
            limit: usize,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<MemoryEntry>> + Send + 'a>>
        {
            Box::pin(async move { self.search(query, limit) })
        }

        fn extract<'a>(
            &'a self,
            assistant_reply: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<NewMemory>> + Send + 'a>>
        {
            Box::pin(async move {
                if !self.auto_extract || assistant_reply.len() < 50 {
                    return None;
                }
                Some(NewMemory {
                    title: assistant_reply[..assistant_reply.len().min(100)].to_string(),
                    content: assistant_reply.to_string(),
                    tags: Some("auto-extracted".to_string()),
                })
            })
        }

        fn save<'a>(
            &'a self,
            memory: &'a NewMemory,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.save_entry(&memory.title, &memory.content, memory.tags.as_deref());
            })
        }
    }
}

#[cfg(feature = "storage")]
pub use storage_impl::MemoryStore;

// ── Hook bridge ──

/// Bridges [`MemoryProvider`] into an [`AgentHook`].
///
/// Before each LLM call, retrieves relevant memories and injects them as a system message.
/// After each LLM call, extracts and saves new memories via the provider.
pub struct MemoryHook {
    provider: Arc<dyn MemoryProvider>,
    max_injected: usize,
}

impl MemoryHook {
    pub fn new(provider: Arc<dyn MemoryProvider>, max_injected: usize) -> Self {
        Self {
            provider,
            max_injected,
        }
    }
}

#[async_trait]
impl AgentHook for MemoryHook {
    async fn before_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        messages: &mut Vec<Message>,
    ) -> HookResult {
        let query = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        if query.is_empty() {
            return HookResult::Continue;
        }

        match self.provider.retrieve(query, self.max_injected).await {
            memories if !memories.is_empty() => {
                let mem_text: Vec<String> = memories
                    .iter()
                    .map(|m| format!("- {} (weight: {:.2})", m.content, m.weight))
                    .collect();
                let injection = format!(
                    "<relevant-memories>\n{}\n</relevant-memories>",
                    mem_text.join("\n")
                );
                if let Some(pos) = messages.iter().position(|m| m.role == Role::User) {
                    messages.insert(pos, Message::system(injection));
                }
            }
            _ => {}
        }

        HookResult::Continue
    }

    async fn after_llm_call(
        &self,
        _ctx: &HookContext<'_>,
        response: &mut AgentResponse,
    ) -> HookResult {
        let content = match &response.kind {
            AgentResponseKind::MessageComplete(msg) => &msg.content,
            _ => return HookResult::Continue,
        };

        if let Some(new_mem) = self.provider.extract(content).await {
            self.provider.save(&new_mem).await;
        }

        HookResult::Continue
    }
}
