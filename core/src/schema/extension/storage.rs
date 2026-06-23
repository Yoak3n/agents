use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::schema::common::{Message, Role};

pub struct Storage {
    conn: Mutex<Connection>,
}

impl Storage {
    pub fn new() -> Result<Self, StorageError> {
        let db_path = Self::db_path()?;
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_tables()?;
        Ok(storage)
    }

    pub fn with_path(path: &std::path::Path) -> Result<Self, StorageError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.init_tables()?;
        Ok(storage)
    }

    fn db_path() -> Result<PathBuf, StorageError> {
        let base = dirs::data_dir()
            .or_else(dirs::config_dir)
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(base.join("ai-partner").join("conversations.db"))
    }

    fn init_tables(&self) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                title           TEXT,
                first_message   TEXT,
                pinned          INTEGER NOT NULL DEFAULT 0,
                archived        INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);

            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                sort_order      INTEGER NOT NULL,
                reasoning_content TEXT,
                tool_calls      TEXT,
                tool_call_id    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, sort_order);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, sort_order);

            CREATE TABLE IF NOT EXISTS summaries (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                content         TEXT NOT NULL,
                message_range   TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_conversation ON summaries(conversation_id);

            CREATE TABLE IF NOT EXISTS documents (
                id              TEXT PRIMARY KEY,
                summary_id      TEXT NOT NULL REFERENCES summaries(id) ON DELETE CASCADE,
                session_id      TEXT NOT NULL,
                content         TEXT NOT NULL,
                chunk_index     INTEGER NOT NULL,
                token_count     INTEGER NOT NULL,
                embedding       BLOB,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
            CREATE INDEX IF NOT EXISTS idx_documents_summary ON documents(summary_id);

            CREATE TABLE IF NOT EXISTS memories (
                id                TEXT PRIMARY KEY,
                title             TEXT NOT NULL,
                content           TEXT NOT NULL,
                tags              TEXT,
                session_id        TEXT,
                conversation_id   TEXT,
                weight            REAL NOT NULL DEFAULT 1.0,
                last_activated_at TEXT NOT NULL,
                activation_count  INTEGER NOT NULL DEFAULT 0,
                embedding         BLOB,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_title ON memories(title);
            CREATE INDEX IF NOT EXISTS idx_memories_weight ON memories(weight DESC);",
        )?;
        Ok(())
    }

    // ── Sessions ──

    pub fn create_session(
        &self,
        id: &str,
        title: Option<&str>,
        first_message: Option<&str>,
    ) -> Result<(), StorageError> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "INSERT INTO sessions (id, title, first_message, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, title, first_message, now, now],
        )?;
        Ok(())
    }

    pub fn update_session_first_message(
        &self,
        id: &str,
        first_message: &str,
    ) -> Result<(), StorageError> {
        self.conn.lock().unwrap().execute(
            "UPDATE sessions SET first_message = ?1 WHERE id = ?2 AND first_message IS NULL",
            params![first_message, id],
        )?;
        Ok(())
    }

    pub fn toggle_session_pinned(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions SET pinned = NOT pinned WHERE id = ?1",
            params![id],
        )?;
        let pinned: bool = conn.query_row(
            "SELECT pinned FROM sessions WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(pinned)
    }

    pub fn toggle_session_archived(&self, id: &str) -> Result<bool, StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions SET archived = NOT archived WHERE id = ?1",
            params![id],
        )?;
        let archived: bool = conn.query_row(
            "SELECT archived FROM sessions WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(archived)
    }

    pub fn delete_session(&self, id: &str) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM sessions WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>, StorageError> {
        let conn = self.conn.lock().unwrap();

        // 先清理没有消息的空 session
        conn.execute(
            "DELETE FROM sessions WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)",
            [],
        )?;

        let mut stmt = conn.prepare(
            "SELECT s.id, s.title, s.first_message, s.pinned, s.archived,
                    s.created_at, s.updated_at, COUNT(m.id)
             FROM sessions s
             INNER JOIN messages m ON m.session_id = s.id
             GROUP BY s.id
             ORDER BY s.pinned DESC, s.updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                first_message: row.get(2)?,
                pinned: row.get::<_, i32>(3)? != 0,
                archived: row.get::<_, i32>(4)? != 0,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
                message_count: row.get(7)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ── Conversations (within a session) ──

    pub fn create_conversation(&self, id: &str, session_id: &str) -> Result<(), StorageError> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "INSERT INTO conversations (id, session_id, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, session_id, now, now],
        )?;
        Ok(())
    }

    pub fn list_conversations(
        &self,
        session_id: &str,
    ) -> Result<Vec<ConversationInfo>, StorageError> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT c.id, c.session_id, c.created_at, c.updated_at, COUNT(m.id)
             FROM conversations c
             LEFT JOIN messages m ON m.conversation_id = c.id
             WHERE c.session_id = ?1
             GROUP BY c.id
             ORDER BY c.created_at ASC",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok(ConversationInfo {
                id: row.get(0)?,
                session_id: row.get(1)?,
                created_at: row.get(2)?,
                updated_at: row.get(3)?,
                message_count: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ── Messages ──

    pub fn save_message(
        &self,
        session_id: &str,
        conversation_id: &str,
        msg: &Message,
        sort_order: i64,
    ) -> Result<(), StorageError> {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };
        let tool_calls_json = msg
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap_or_default());
        self.conn.lock().unwrap().execute(
            "INSERT INTO messages (id, session_id, conversation_id, role, content, timestamp, sort_order, reasoning_content, tool_calls, tool_call_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET content=excluded.content, reasoning_content=excluded.reasoning_content",
            params![
                msg.id.to_string(),
                session_id,
                conversation_id,
                role_str,
                msg.content,
                msg.timestamp.to_rfc3339(),
                sort_order,
                msg.reasoning_content,
                tool_calls_json,
                msg.tool_call_id,
            ],
        )?;
        // touch session updated_at
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![now, session_id],
        )?;
        Ok(())
    }

    pub fn load_messages(&self, session_id: &str) -> Result<Vec<Message>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, role, content, timestamp, reasoning_content, tool_calls, tool_call_id FROM messages
             WHERE session_id = ?1 ORDER BY sort_order",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            let id_str: String = row.get(0)?;
            let role_str: String = row.get(1)?;
            let content: String = row.get(2)?;
            let ts_str: String = row.get(3)?;
            let reasoning_content: Option<String> = row.get(4)?;
            let tool_calls_json: Option<String> = row.get(5)?;
            let tool_call_id: Option<String> = row.get(6)?;

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => Role::Tool,
            };
            let timestamp = chrono::DateTime::parse_from_rfc3339(&ts_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            let tool_calls = tool_calls_json.and_then(|json| serde_json::from_str(&json).ok());

            Ok(Message {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                role,
                content,
                timestamp,
                reasoning_content,
                tool_calls,
                tool_call_id,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Load messages for a specific conversation (ordered by sort_order).
    pub fn load_messages_by_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<Message>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, role, content, timestamp, reasoning_content, tool_calls, tool_call_id FROM messages
             WHERE conversation_id = ?1 ORDER BY sort_order",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
            let id_str: String = row.get(0)?;
            let role_str: String = row.get(1)?;
            let content: String = row.get(2)?;
            let ts_str: String = row.get(3)?;
            let reasoning_content: Option<String> = row.get(4)?;
            let tool_calls_json: Option<String> = row.get(5)?;
            let tool_call_id: Option<String> = row.get(6)?;

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => Role::Tool,
            };
            let timestamp = chrono::DateTime::parse_from_rfc3339(&ts_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            let tool_calls = tool_calls_json.and_then(|json| serde_json::from_str(&json).ok());

            Ok(Message {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                role,
                content,
                timestamp,
                reasoning_content,
                tool_calls,
                tool_call_id,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Load the most recent N messages for a session (ordered by sort_order).
    pub fn load_messages_recent(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Message>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, role, content, timestamp, reasoning_content, tool_calls, tool_call_id FROM messages
             WHERE session_id = ?1 ORDER BY sort_order DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![session_id, limit as i64], |row| {
            let id_str: String = row.get(0)?;
            let role_str: String = row.get(1)?;
            let content: String = row.get(2)?;
            let ts_str: String = row.get(3)?;
            let reasoning_content: Option<String> = row.get(4)?;
            let tool_calls_json: Option<String> = row.get(5)?;
            let tool_call_id: Option<String> = row.get(6)?;

            let role = match role_str.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => Role::Tool,
            };
            let timestamp = chrono::DateTime::parse_from_rfc3339(&ts_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now());
            let tool_calls = tool_calls_json.and_then(|json| serde_json::from_str(&json).ok());

            Ok(Message {
                id: uuid::Uuid::parse_str(&id_str).unwrap_or_else(|_| uuid::Uuid::new_v4()),
                role,
                content,
                timestamp,
                reasoning_content,
                tool_calls,
                tool_call_id,
            })
        })?;
        let mut messages: Vec<Message> = rows.collect::<Result<Vec<_>, _>>()?;
        messages.reverse(); // DESC → chronological order
        Ok(messages)
    }

    /// Get the total number of messages in a session.
    pub fn get_message_count(&self, session_id: &str) -> Result<usize, StorageError> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    // ── Summaries ──

    pub fn save_summary(
        &self,
        conversation_id: &str,
        content: &str,
        message_range: &str,
    ) -> Result<String, StorageError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "INSERT INTO summaries (id, conversation_id, content, message_range, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, conversation_id, content, message_range, now],
        )?;
        Ok(id)
    }

    pub fn get_summaries(&self, conversation_id: &str) -> Result<Vec<Summary>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, content, message_range, created_at
             FROM summaries WHERE conversation_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
            Ok(Summary {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                content: row.get(2)?,
                message_range: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Search summaries by keyword across all conversations.
    pub fn search_summaries(
        &self,
        query: &str,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<Summary>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, content, message_range, created_at
             FROM summaries WHERE content LIKE ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
        )?;
        let pattern = format!("%{query}%");
        let rows = stmt.query_map(params![pattern, limit, offset], |row| {
            Ok(Summary {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                content: row.get(2)?,
                message_range: row.get(3)?,
                created_at: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ── Documents (RAG) ──

    pub fn save_document(
        &self,
        summary_id: &str,
        session_id: &str,
        content: &str,
        chunk_index: i32,
        token_count: i32,
    ) -> Result<String, StorageError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "INSERT INTO documents (id, summary_id, session_id, content, chunk_index, token_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, summary_id, session_id, content, chunk_index, token_count, now],
        )?;
        Ok(id)
    }

    pub fn save_document_embedding(
        &self,
        document_id: &str,
        embedding: &[f32],
    ) -> Result<(), StorageError> {
        let bytes = f32_vec_to_bytes(embedding);
        self.conn.lock().unwrap().execute(
            "UPDATE documents SET embedding = ?1 WHERE id = ?2",
            params![bytes, document_id],
        )?;
        Ok(())
    }

    pub fn get_documents_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<Document>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, summary_id, session_id, content, chunk_index, token_count, embedding, created_at
             FROM documents WHERE session_id = ?1 ORDER BY chunk_index",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            let embedding_bytes: Option<Vec<u8>> = row.get(6)?;
            Ok(Document {
                id: row.get(0)?,
                summary_id: row.get(1)?,
                session_id: row.get(2)?,
                content: row.get(3)?,
                chunk_index: row.get(4)?,
                token_count: row.get(5)?,
                embedding: embedding_bytes.map(|b| bytes_to_f32_vec(&b)),
                created_at: row.get(7)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // ── Memories (weighted with forgetting curve) ──

    const SELECT_MEMORY_COLS: &'static str = "id, title, content, tags, session_id, conversation_id, weight, last_activated_at, activation_count, created_at, updated_at";

    fn row_to_memory(row: &rusqlite::Row) -> rusqlite::Result<MemoryEntry> {
        Ok(MemoryEntry {
            id: row.get(0)?,
            title: row.get(1)?,
            content: row.get(2)?,
            tags: row.get(3)?,
            session_id: row.get(4)?,
            conversation_id: row.get(5)?,
            weight: row.get(6)?,
            last_activated_at: row.get(7)?,
            activation_count: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
        })
    }

    pub fn save_memory(
        &self,
        id: Option<&str>,
        title: &str,
        content: &str,
        tags: Option<&str>,
        session_id: Option<&str>,
        conversation_id: Option<&str>,
    ) -> Result<String, StorageError> {
        let now = chrono::Utc::now().to_rfc3339();
        let id = match id {
            Some(existing_id) => {
                self.conn.lock().unwrap().execute(
                    "INSERT INTO memories (id, title, content, tags, session_id, conversation_id, weight, last_activated_at, activation_count, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1.0, ?7, 0, ?7, ?7)
                     ON CONFLICT(id) DO UPDATE SET title=excluded.title, content=excluded.content, tags=excluded.tags, session_id=excluded.session_id, conversation_id=excluded.conversation_id, updated_at=excluded.updated_at",
                    params![existing_id, title, content, tags, session_id, conversation_id, now],
                )?;
                existing_id.to_string()
            }
            None => {
                let new_id = uuid::Uuid::new_v4().to_string();
                self.conn.lock().unwrap().execute(
                    "INSERT INTO memories (id, title, content, tags, session_id, conversation_id, weight, last_activated_at, activation_count, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1.0, ?7, 0, ?7, ?7)",
                    params![new_id, title, content, tags, session_id, conversation_id, now],
                )?;
                new_id
            }
        };
        Ok(id)
    }

    /// Calculate weight decay factor based on days since last activation.
    fn calc_decay_weight(weight: f64, last_activated_at: &str, decay_rate: f64) -> f64 {
        let now = chrono::Utc::now();
        if let Ok(last) = chrono::DateTime::parse_from_rfc3339(last_activated_at) {
            let last_utc = last.with_timezone(&chrono::Utc);
            let days = (now - last_utc).num_days().max(0) as f64;
            // weight * (1.0 - decay_rate) ^ days
            weight * (1.0 - decay_rate).powf(days)
        } else {
            weight
        }
    }

    /// Read a memory entry, applying real-time weight decay first.
    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryEntry>, StorageError> {
        let conn = self.conn.lock().unwrap();
        // First get the current weight and last_activated_at
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories WHERE id = ?1",
            Self::SELECT_MEMORY_COLS
        ))?;
        let mut rows = stmt.query_map(params![id], Self::row_to_memory)?;
        match rows.next() {
            Some(row) => {
                let mut entry = row?;
                let decayed = Self::calc_decay_weight(
                    entry.weight,
                    &entry.last_activated_at,
                    MEMORY_DECAY_RATE,
                );
                // Update weight in DB
                conn.execute(
                    "UPDATE memories SET weight = ?2 WHERE id = ?1",
                    params![id, decayed],
                )?;
                entry.weight = decayed;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Activate a memory: decay, boost weight (+0.1, cap 2.0), bump activation count, return entry.
    pub fn activate_memory(&self, id: &str) -> Result<Option<MemoryEntry>, StorageError> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        // Get current memory
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories WHERE id = ?1",
            Self::SELECT_MEMORY_COLS
        ))?;
        let mut rows = stmt.query_map(params![id], Self::row_to_memory)?;
        match rows.next() {
            Some(row) => {
                let entry = row?;
                let decayed = Self::calc_decay_weight(
                    entry.weight,
                    &entry.last_activated_at,
                    MEMORY_DECAY_RATE,
                );
                let new_weight = (decayed + 0.1_f64).min(2.0_f64);
                conn.execute(
                    "UPDATE memories SET weight = ?2, last_activated_at = ?3, activation_count = activation_count + 1 WHERE id = ?1",
                    params![id, new_weight, now],
                )?;
                // Fetch updated entry
                let mut stmt2 = conn.prepare(&format!(
                    "SELECT {} FROM memories WHERE id = ?1",
                    Self::SELECT_MEMORY_COLS
                ))?;
                let mut rows2 = stmt2.query_map(params![id], Self::row_to_memory)?;
                match rows2.next() {
                    Some(row) => Ok(Some(row?)),
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    /// List memories with real-time weight decay, paginated.
    pub fn list_memories(&self, offset: i64, limit: i64) -> Result<Vec<MemoryEntry>, StorageError> {
        let conn = self.conn.lock().unwrap();
        // Fetch all memories and apply decay in Rust
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories ORDER BY weight DESC, updated_at DESC LIMIT ?1 OFFSET ?2",
            Self::SELECT_MEMORY_COLS
        ))?;
        let rows = stmt.query_map(params![limit, offset], Self::row_to_memory)?;
        let mut entries: Vec<MemoryEntry> = Vec::new();
        for row in rows {
            let mut entry = row?;
            let decayed =
                Self::calc_decay_weight(entry.weight, &entry.last_activated_at, MEMORY_DECAY_RATE);
            // Update weight in DB
            conn.execute(
                "UPDATE memories SET weight = ?2 WHERE id = ?1",
                params![entry.id, decayed],
            )?;
            entry.weight = decayed;
            entries.push(entry);
        }
        // Re-sort by decayed weight
        entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(entries)
    }

    /// Search memories by text with real-time decay, paginated.
    pub fn search_memories(
        &self,
        query: &str,
        offset: i64,
        limit: i64,
    ) -> Result<Vec<MemoryEntry>, StorageError> {
        let pattern = format!("%{query}%");
        let conn = self.conn.lock().unwrap();
        // Fetch matching memories and apply decay in Rust
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM memories WHERE title LIKE ?1 OR content LIKE ?1 OR tags LIKE ?1 ORDER BY weight DESC LIMIT ?2 OFFSET ?3",
            Self::SELECT_MEMORY_COLS
        ))?;
        let rows = stmt.query_map(params![pattern, limit, offset], Self::row_to_memory)?;
        let mut entries: Vec<MemoryEntry> = Vec::new();
        for row in rows {
            let mut entry = row?;
            let decayed =
                Self::calc_decay_weight(entry.weight, &entry.last_activated_at, MEMORY_DECAY_RATE);
            conn.execute(
                "UPDATE memories SET weight = ?2 WHERE id = ?1",
                params![entry.id, decayed],
            )?;
            entry.weight = decayed;
            entries.push(entry);
        }
        entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(entries)
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, StorageError> {
        let rows = self
            .conn
            .lock()
            .unwrap()
            .execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    pub fn save_memory_embedding(&self, id: &str, embedding: &[f32]) -> Result<(), StorageError> {
        let bytes = f32_vec_to_bytes(embedding);
        self.conn.lock().unwrap().execute(
            "UPDATE memories SET embedding = ?2 WHERE id = ?1",
            params![id, bytes],
        )?;
        Ok(())
    }

    pub fn search_memories_embedding(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {}, embedding FROM memories WHERE embedding IS NOT NULL",
            Self::SELECT_MEMORY_COLS
        ))?;

        let mut results: Vec<(MemoryEntry, f32)> = Vec::new();
        let rows = stmt.query_map([], |row| {
            let entry = Self::row_to_memory(row)?;
            let embedding_bytes: Vec<u8> = row.get(11)?;
            Ok((entry, embedding_bytes))
        })?;

        for row in rows.flatten() {
            let mut entry = row.0;
            let mem_embedding = bytes_to_f32_vec(&row.1);
            let score = cosine_similarity(query_embedding, &mem_embedding);
            let decayed =
                Self::calc_decay_weight(entry.weight, &entry.last_activated_at, MEMORY_DECAY_RATE);
            entry.weight = decayed;
            results.push((entry, score));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().take(limit).map(|(e, _)| e).collect())
    }

    /// Vector similarity search (cosine similarity)
    pub fn search_documents(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<DocumentSearchResult>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, content, chunk_index, embedding
             FROM documents WHERE embedding IS NOT NULL",
        )?;

        let mut results: Vec<DocumentSearchResult> = Vec::new();
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let sess_id: String = row.get(1)?;
            let content: String = row.get(2)?;
            let chunk_index: i32 = row.get(3)?;
            let embedding_bytes: Vec<u8> = row.get(4)?;
            Ok((id, sess_id, content, chunk_index, embedding_bytes))
        })?;

        for row in rows.flatten() {
            let doc_embedding = bytes_to_f32_vec(&row.4);
            let score = cosine_similarity(query_embedding, &doc_embedding);
            results.push(DocumentSearchResult {
                document_id: row.0,
                session_id: row.1,
                content: row.2,
                chunk_index: row.3,
                score,
            });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }
}

// ── Vector utility functions ──

fn f32_vec_to_bytes(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub id: String,
    pub conversation_id: String,
    pub content: String,
    pub message_range: String,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub summary_id: String,
    pub session_id: String,
    pub content: String,
    pub chunk_index: i32,
    pub token_count: i32,
    pub embedding: Option<Vec<f32>>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct DocumentSearchResult {
    pub document_id: String,
    pub session_id: String,
    pub content: String,
    pub chunk_index: i32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: Option<String>,
    pub first_message: Option<String>,
    pub pinned: bool,
    pub archived: bool,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationInfo {
    pub id: String,
    pub session_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub title: String,
    pub content: String,
    pub tags: Option<String>,
    pub session_id: Option<String>,
    pub conversation_id: Option<String>,
    pub weight: f64,
    pub last_activated_at: String,
    pub activation_count: i64,
    pub created_at: String,
    pub updated_at: String,
}

/// Daily decay rate for memory weight (exponential forgetting curve).
const MEMORY_DECAY_RATE: f64 = 0.05;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_storage() -> Storage {
        let dir = std::env::temp_dir().join(format!("ai-partner-test-{}", uuid::Uuid::new_v4()));
        Storage::with_path(&dir.join("test.db")).unwrap()
    }

    #[test]
    fn test_session_and_messages() {
        let storage = temp_storage();
        let session_id = "test-session-1";
        let conv_id = "test-conv-1";
        storage
            .create_session(session_id, Some("Test Chat"), None)
            .unwrap();
        storage.create_conversation(conv_id, session_id).unwrap();

        let msg1 = Message::user("Hello");
        let msg2 = Message::assistant("Hi there!");
        storage.save_message(session_id, conv_id, &msg1, 0).unwrap();
        storage.save_message(session_id, conv_id, &msg2, 1).unwrap();

        let messages = storage.load_messages(session_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].content, "Hi there!");

        let sessions = storage.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].message_count, 2);

        storage.delete_session(session_id).unwrap();
        assert!(storage.list_sessions().unwrap().is_empty());
    }
}
