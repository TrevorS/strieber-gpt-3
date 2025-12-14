//! SQLite storage implementations for responses and conversations.
//!
//! Provides persistent storage using SQLite with sqlx.
//! Converts between StoredResponse/StoredConversation (with Instant/Duration) and database timestamps.

use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sqlx::Row;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use tokio::runtime::Handle;
use tracing::error;

use crate::models::{
    Conversation, ConversationItem, ConversationItemContent, CreateResponseRequest, InputItem,
    ListResponse, Metadata, OutputStatus, PaginationQuery, Response, SortOrder,
};
use crate::state::{ConversationStore, ResponseStore, StoredResponse};
use crate::storage::memory::StoredConversation;
use crate::translation::{conversation_id, function_call_id, item_id, message_id, reasoning_id};

/// SQLite-based response store.
#[derive(Clone)]
pub struct SqliteResponseStore {
    pool: SqlitePool,
}

impl SqliteResponseStore {
    /// Create a new SQLite store at the given path.
    /// Runs migrations on startup.
    pub async fn new(path: &Path) -> Result<Self, sqlx::Error> {
        // Create connection options
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);

        // Create connection pool
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        // Run migrations
        sqlx::migrate!("./migrations").run(&pool).await?;

        Ok(Self { pool })
    }

    /// Create a new SQLite store from an existing pool.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Convert Instant to Unix timestamp.
    /// Uses the elapsed time since the Instant was created.
    fn instant_to_timestamp(instant: Instant) -> i64 {
        let elapsed = instant.elapsed();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now - elapsed.as_secs() as i64
    }

    /// Convert Unix timestamp to Instant.
    /// Creates an Instant by calculating how long ago the timestamp was.
    fn timestamp_to_instant(timestamp: i64) -> Instant {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let seconds_ago = now - timestamp;
        if seconds_ago > 0 {
            Instant::now() - Duration::from_secs(seconds_ago as u64)
        } else {
            Instant::now()
        }
    }

    /// Convert Duration to seconds for storage.
    fn duration_to_seconds(duration: Duration) -> i64 {
        duration.as_secs() as i64
    }

    /// Convert seconds to Duration.
    fn seconds_to_duration(seconds: i64) -> Duration {
        Duration::from_secs(seconds.max(0) as u64)
    }
}

impl ResponseStore for SqliteResponseStore {
    fn store(&self, response: Response, request: CreateResponseRequest) {
        let pool = self.pool.clone();
        let id = response.id.clone();

        // Use block_in_place to bridge sync trait to async implementation
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                // Serialize response and request to JSON
                let response_json = serde_json::to_string(&response).unwrap_or_else(|e| {
                    tracing::error!("Failed to serialize response: {}", e);
                    "{}".to_string()
                });
                let request_json = serde_json::to_string(&request).unwrap_or_else(|e| {
                    tracing::error!("Failed to serialize request: {}", e);
                    "{}".to_string()
                });

                // Calculate timestamps
                let created_at = Self::instant_to_timestamp(Instant::now());
                let ttl_seconds = Self::duration_to_seconds(Duration::from_secs(3600)); // 1 hour default
                let expires_at = created_at + ttl_seconds;

                // Insert or replace response
                let result = sqlx::query(
                    r#"
                    INSERT OR REPLACE INTO responses (id, response_json, request_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    "#,
                )
                .bind(&id)
                .bind(&response_json)
                .bind(&request_json)
                .bind(created_at)
                .bind(expires_at)
                .execute(&pool)
                .await;

                if let Err(e) = result {
                    tracing::error!("Failed to store response {}: {}", id, e);
                }
            })
        });
    }

    fn get(&self, id: &str) -> Option<StoredResponse> {
        let pool = self.pool.clone();
        let id = id.to_string();

        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                let result = sqlx::query(
                    r#"
                    SELECT response_json, request_json, created_at, expires_at
                    FROM responses
                    WHERE id = ?
                    "#,
                )
                .bind(&id)
                .fetch_optional(&pool)
                .await;

                match result {
                    Ok(Some(row)) => {
                        // Extract all values from row first
                        let response_json: String =
                            row.try_get("response_json").unwrap_or_default();
                        let request_json: String = row.try_get("request_json").unwrap_or_default();
                        let created_at: i64 = row.try_get("created_at").unwrap_or(0);
                        let expires_at: i64 = row.try_get("expires_at").unwrap_or(0);

                        // Drop row to release borrow
                        drop(row);

                        // Deserialize from JSON strings
                        // Note: Response has a &'static str field which requires special handling
                        // We leak the string to make it 'static - this is acceptable for long-lived data
                        let response: Response =
                            match serde_json::from_str(Box::leak(response_json.into_boxed_str())) {
                                Ok(r) => r,
                                Err(e) => {
                                    tracing::error!("Failed to deserialize response {}: {}", id, e);
                                    return None;
                                }
                            };

                        let request: CreateResponseRequest =
                            match serde_json::from_str(&request_json) {
                                Ok(r) => r,
                                Err(e) => {
                                    tracing::error!("Failed to deserialize request {}: {}", id, e);
                                    return None;
                                }
                            };

                        // Convert timestamps back to Instant/Duration
                        let created_instant = Self::timestamp_to_instant(created_at);
                        let ttl = Self::seconds_to_duration(expires_at - created_at);

                        Some(StoredResponse {
                            response,
                            request,
                            created_at: created_instant,
                            ttl,
                        })
                    }
                    Ok(None) => None,
                    Err(e) => {
                        tracing::error!("Failed to fetch response {}: {}", id, e);
                        None
                    }
                }
            })
        })
    }

    fn delete(&self, id: &str) -> bool {
        let pool = self.pool.clone();
        let id = id.to_string();

        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                let result = sqlx::query(
                    r#"
                    DELETE FROM responses WHERE id = ?
                    "#,
                )
                .bind(&id)
                .execute(&pool)
                .await;

                match result {
                    Ok(result) => result.rows_affected() > 0,
                    Err(e) => {
                        tracing::error!("Failed to delete response {}: {}", id, e);
                        false
                    }
                }
            })
        })
    }

    fn list_ids(&self) -> Vec<String> {
        let pool = self.pool.clone();

        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                let result = sqlx::query(
                    r#"
                    SELECT id FROM responses ORDER BY created_at DESC
                    "#,
                )
                .fetch_all(&pool)
                .await;

                match result {
                    Ok(rows) => rows.iter().map(|row| row.get("id")).collect(),
                    Err(e) => {
                        tracing::error!("Failed to list response IDs: {}", e);
                        Vec::new()
                    }
                }
            })
        })
    }

    fn cleanup_expired(&self) {
        let pool = self.pool.clone();

        tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;

                let result = sqlx::query(
                    r#"
                    DELETE FROM responses WHERE expires_at < ?
                    "#,
                )
                .bind(now)
                .execute(&pool)
                .await;

                match result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            tracing::info!(
                                "Cleaned up {} expired responses",
                                result.rows_affected()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to cleanup expired responses: {}", e);
                    }
                }
            })
        });
    }
}

// ============================================================================
// SqliteConversationStore Implementation
// ============================================================================

/// SQLite-backed conversation store.
#[derive(Clone)]
pub struct SqliteConversationStore {
    pool: SqlitePool,
}

impl SqliteConversationStore {
    /// Create a new SQLite conversation store with the given pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    async fn list_async(&self, query: &PaginationQuery) -> ListResponse<Conversation> {
        // Build ORDER BY clause
        let order_by = match query.order {
            SortOrder::Desc => "created_at DESC",
            SortOrder::Asc => "created_at ASC",
        };

        // If we have a cursor, we need to find the position after it
        let conversations = if let Some(ref after) = query.after {
            // Get the created_at timestamp of the cursor conversation
            let cursor_result = sqlx::query("SELECT created_at FROM conversations WHERE id = ?")
                .bind(after)
                .fetch_optional(&self.pool)
                .await;

            match cursor_result {
                Ok(Some(row)) => {
                    let cursor_created_at: i64 = row.get("created_at");

                    // Fetch conversations based on the order
                    let sql = match query.order {
                        SortOrder::Desc => format!(
                            "SELECT id, metadata_json, created_at FROM conversations
                             WHERE created_at < ? OR (created_at = ? AND id > ?)
                             ORDER BY {} LIMIT ?",
                            order_by
                        ),
                        SortOrder::Asc => format!(
                            "SELECT id, metadata_json, created_at FROM conversations
                             WHERE created_at > ? OR (created_at = ? AND id > ?)
                             ORDER BY {} LIMIT ?",
                            order_by
                        ),
                    };

                    sqlx::query(&sql)
                        .bind(cursor_created_at)
                        .bind(cursor_created_at)
                        .bind(after)
                        .bind(query.limit + 1) // Fetch one extra to check has_more
                        .fetch_all(&self.pool)
                        .await
                }
                Ok(None) => {
                    // Cursor not found, return empty
                    return ListResponse::empty();
                }
                Err(e) => {
                    error!("Failed to get cursor conversation: {}", e);
                    return ListResponse::empty();
                }
            }
        } else {
            // No cursor, fetch from beginning
            let sql = format!(
                "SELECT id, metadata_json, created_at FROM conversations ORDER BY {} LIMIT ?",
                order_by
            );

            sqlx::query(&sql)
                .bind(query.limit + 1) // Fetch one extra to check has_more
                .fetch_all(&self.pool)
                .await
        };

        match conversations {
            Ok(rows) => {
                let mut data: Vec<Conversation> = Vec::new();
                let has_more = rows.len() > query.limit as usize;
                let rows_to_take = rows.len().min(query.limit as usize);

                for row in rows.into_iter().take(rows_to_take) {
                    let id: String = row.get("id");
                    let metadata_json: Option<String> = row.get("metadata_json");
                    let created_at: i64 = row.get("created_at");

                    let metadata = if let Some(json) = metadata_json {
                        match serde_json::from_str::<Metadata>(&json) {
                            Ok(m) => Some(m),
                            Err(e) => {
                                error!("Failed to deserialize metadata for {}: {}", id, e);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    data.push(Conversation {
                        id,
                        object: Conversation::OBJECT,
                        created_at,
                        metadata,
                    });
                }

                let first_id = data.first().map(|c| c.id.clone());
                let last_id = data.last().map(|c| c.id.clone());

                ListResponse::new(data, has_more).with_ids(first_id, last_id)
            }
            Err(e) => {
                error!("Failed to list conversations: {}", e);
                ListResponse::empty()
            }
        }
    }

    async fn create_async(
        &self,
        metadata: Option<Metadata>,
        items: Option<Vec<InputItem>>,
    ) -> Conversation {
        let id = conversation_id();
        let created_at = chrono::Utc::now().timestamp();

        let metadata_json = metadata.as_ref().and_then(|m| {
            serde_json::to_string(m)
                .map_err(|e| {
                    error!("Failed to serialize metadata: {}", e);
                    e
                })
                .ok()
        });

        // Insert conversation
        let result = sqlx::query(
            "INSERT INTO conversations (id, metadata_json, created_at, expires_at) VALUES (?, ?, ?, ?)"
        )
        .bind(&id)
        .bind(&metadata_json)
        .bind(created_at)
        .bind(Option::<i64>::None) // No expiration by default
        .execute(&self.pool)
        .await;

        if let Err(e) = result {
            error!("Failed to create conversation {}: {}", id, e);
            // Return the conversation object anyway, as the in-memory version does
        }

        // Insert items if provided
        if let Some(items) = items {
            for (order, item) in items.into_iter().enumerate() {
                let item = input_to_conversation_item(item);
                let content_json = match serde_json::to_string(&item) {
                    Ok(json) => json,
                    Err(e) => {
                        error!("Failed to serialize item: {}", e);
                        continue;
                    }
                };

                let result = sqlx::query(
                    "INSERT INTO conversation_items (id, conversation_id, item_order, content_json, created_at)
                     VALUES (?, ?, ?, ?, ?)"
                )
                .bind(&item.id)
                .bind(&id)
                .bind(order as i64)
                .bind(&content_json)
                .bind(created_at)
                .execute(&self.pool)
                .await;

                if let Err(e) = result {
                    error!(
                        "Failed to insert item {} for conversation {}: {}",
                        item.id, id, e
                    );
                }
            }
        }

        Conversation {
            id,
            object: Conversation::OBJECT,
            created_at,
            metadata,
        }
    }

    async fn get_async(&self, id: &str) -> Option<Conversation> {
        let result =
            sqlx::query("SELECT id, metadata_json, created_at FROM conversations WHERE id = ?")
                .bind(id)
                .fetch_optional(&self.pool)
                .await;

        match result {
            Ok(Some(row)) => {
                let id: String = row.get("id");
                let metadata_json: Option<String> = row.get("metadata_json");
                let created_at: i64 = row.get("created_at");

                let metadata = if let Some(json) = metadata_json {
                    match serde_json::from_str::<Metadata>(&json) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            error!("Failed to deserialize metadata for {}: {}", id, e);
                            None
                        }
                    }
                } else {
                    None
                };

                Some(Conversation {
                    id,
                    object: Conversation::OBJECT,
                    created_at,
                    metadata,
                })
            }
            Ok(None) => None,
            Err(e) => {
                error!("Failed to get conversation {}: {}", id, e);
                None
            }
        }
    }

    async fn update_async(&self, id: &str, metadata: Metadata) -> Option<Conversation> {
        let metadata_json = match serde_json::to_string(&metadata) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize metadata: {}", e);
                return None;
            }
        };

        let result = sqlx::query("UPDATE conversations SET metadata_json = ? WHERE id = ?")
            .bind(&metadata_json)
            .bind(id)
            .execute(&self.pool)
            .await;

        match result {
            Ok(res) if res.rows_affected() > 0 => self.get_async(id).await,
            Ok(_) => None,
            Err(e) => {
                error!("Failed to update conversation {}: {}", id, e);
                None
            }
        }
    }

    async fn delete_async(&self, id: &str) -> bool {
        let result = sqlx::query("DELETE FROM conversations WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await;

        match result {
            Ok(res) => res.rows_affected() > 0,
            Err(e) => {
                error!("Failed to delete conversation {}: {}", id, e);
                false
            }
        }
    }

    async fn list_items_async(
        &self,
        id: &str,
        query: &PaginationQuery,
    ) -> Option<ListResponse<ConversationItem>> {
        // First check if conversation exists
        self.get_async(id).await.as_ref()?;

        // Build ORDER BY clause
        let order_by = match query.order {
            SortOrder::Asc => "item_order ASC",
            SortOrder::Desc => "item_order DESC",
        };

        // Fetch items
        let items = if let Some(ref after) = query.after {
            // Get the order of the cursor item
            let cursor_result = sqlx::query(
                "SELECT item_order FROM conversation_items WHERE id = ? AND conversation_id = ?",
            )
            .bind(after)
            .bind(id)
            .fetch_optional(&self.pool)
            .await;

            match cursor_result {
                Ok(Some(row)) => {
                    let cursor_order: i64 = row.get("item_order");

                    // Fetch items after cursor
                    let sql = match query.order {
                        SortOrder::Asc => format!(
                            "SELECT id, content_json, item_order FROM conversation_items
                             WHERE conversation_id = ? AND item_order > ?
                             ORDER BY {} LIMIT ?",
                            order_by
                        ),
                        SortOrder::Desc => format!(
                            "SELECT id, content_json, item_order FROM conversation_items
                             WHERE conversation_id = ? AND item_order < ?
                             ORDER BY {} LIMIT ?",
                            order_by
                        ),
                    };

                    sqlx::query(&sql)
                        .bind(id)
                        .bind(cursor_order)
                        .bind(query.limit + 1)
                        .fetch_all(&self.pool)
                        .await
                }
                Ok(None) => {
                    // Cursor not found
                    return Some(ListResponse::empty());
                }
                Err(e) => {
                    error!("Failed to get cursor item: {}", e);
                    return Some(ListResponse::empty());
                }
            }
        } else {
            // No cursor, fetch from beginning
            let sql = format!(
                "SELECT id, content_json, item_order FROM conversation_items
                 WHERE conversation_id = ? ORDER BY {} LIMIT ?",
                order_by
            );

            sqlx::query(&sql)
                .bind(id)
                .bind(query.limit + 1)
                .fetch_all(&self.pool)
                .await
        };

        match items {
            Ok(rows) => {
                let mut data: Vec<ConversationItem> = Vec::new();
                let has_more = rows.len() > query.limit as usize;
                let rows_to_take = rows.len().min(query.limit as usize);

                for row in rows.into_iter().take(rows_to_take) {
                    let content_json: String = row.get("content_json");

                    match serde_json::from_str::<ConversationItem>(&content_json) {
                        Ok(item) => data.push(item),
                        Err(e) => {
                            error!("Failed to deserialize item: {}", e);
                        }
                    }
                }

                let first_id = data.first().map(|i| i.id.clone());
                let last_id = data.last().map(|i| i.id.clone());

                Some(ListResponse::new(data, has_more).with_ids(first_id, last_id))
            }
            Err(e) => {
                error!("Failed to list items for conversation {}: {}", id, e);
                Some(ListResponse::empty())
            }
        }
    }

    async fn add_items_async(
        &self,
        id: &str,
        items: Vec<InputItem>,
    ) -> Option<ListResponse<ConversationItem>> {
        // Check if conversation exists
        self.get_async(id).await.as_ref()?;

        // Get the current max order
        let max_order_result = sqlx::query(
            "SELECT MAX(item_order) as max_order FROM conversation_items WHERE conversation_id = ?",
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await;

        let start_order = match max_order_result {
            Ok(row) => {
                let max_order: Option<i64> = row.get("max_order");
                max_order.map(|m| m + 1).unwrap_or(0)
            }
            Err(e) => {
                error!("Failed to get max order for conversation {}: {}", id, e);
                return None;
            }
        };

        let created_at = chrono::Utc::now().timestamp();
        let mut new_items = Vec::new();

        for (offset, item) in items.into_iter().enumerate() {
            let conv_item = input_to_conversation_item(item);
            let order = start_order + offset as i64;

            let content_json = match serde_json::to_string(&conv_item) {
                Ok(json) => json,
                Err(e) => {
                    error!("Failed to serialize item: {}", e);
                    continue;
                }
            };

            let result = sqlx::query(
                "INSERT INTO conversation_items (id, conversation_id, item_order, content_json, created_at)
                 VALUES (?, ?, ?, ?, ?)"
            )
            .bind(&conv_item.id)
            .bind(id)
            .bind(order)
            .bind(&content_json)
            .bind(created_at)
            .execute(&self.pool)
            .await;

            if let Err(e) = result {
                error!(
                    "Failed to insert item {} for conversation {}: {}",
                    conv_item.id, id, e
                );
                continue;
            }

            new_items.push(conv_item);
        }

        let first_id = new_items.first().map(|i| i.id.clone());
        let last_id = new_items.last().map(|i| i.id.clone());

        Some(ListResponse::new(new_items, false).with_ids(first_id, last_id))
    }

    async fn get_item_async(&self, conv_id: &str, item_id: &str) -> Option<ConversationItem> {
        let result = sqlx::query(
            "SELECT content_json FROM conversation_items WHERE id = ? AND conversation_id = ?",
        )
        .bind(item_id)
        .bind(conv_id)
        .fetch_optional(&self.pool)
        .await;

        match result {
            Ok(Some(row)) => {
                let content_json: String = row.get("content_json");

                match serde_json::from_str::<ConversationItem>(&content_json) {
                    Ok(item) => Some(item),
                    Err(e) => {
                        error!("Failed to deserialize item {}: {}", item_id, e);
                        None
                    }
                }
            }
            Ok(None) => None,
            Err(e) => {
                error!(
                    "Failed to get item {} from conversation {}: {}",
                    item_id, conv_id, e
                );
                None
            }
        }
    }

    async fn delete_item_async(&self, conv_id: &str, item_id: &str) -> Option<Conversation> {
        let result =
            sqlx::query("DELETE FROM conversation_items WHERE id = ? AND conversation_id = ?")
                .bind(item_id)
                .bind(conv_id)
                .execute(&self.pool)
                .await;

        match result {
            Ok(res) if res.rows_affected() > 0 => self.get_async(conv_id).await,
            Ok(_) => None,
            Err(e) => {
                error!(
                    "Failed to delete item {} from conversation {}: {}",
                    item_id, conv_id, e
                );
                None
            }
        }
    }

    async fn get_stored_async(&self, id: &str) -> Option<StoredConversation> {
        // Get conversation
        let conversation = self.get_async(id).await?;

        // Get all items for this conversation
        let items_result = sqlx::query(
            "SELECT content_json FROM conversation_items WHERE conversation_id = ? ORDER BY item_order ASC"
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await;

        let items = match items_result {
            Ok(rows) => {
                let mut items = Vec::new();
                for row in rows {
                    let content_json: String = row.get("content_json");
                    match serde_json::from_str::<ConversationItem>(&content_json) {
                        Ok(item) => items.push(item),
                        Err(e) => {
                            error!("Failed to deserialize item: {}", e);
                        }
                    }
                }
                items
            }
            Err(e) => {
                error!("Failed to get items for conversation {}: {}", id, e);
                vec![]
            }
        };

        // Use defaults for created_at (Instant::now()) and ttl (Duration::MAX)
        Some(StoredConversation {
            conversation,
            items,
            created_at: Instant::now(),
            ttl: Duration::MAX,
        })
    }

    async fn append_output_items_async(&self, id: &str, items: Vec<ConversationItem>) {
        // Get the current max order
        let max_order_result = sqlx::query(
            "SELECT MAX(item_order) as max_order FROM conversation_items WHERE conversation_id = ?",
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await;

        let start_order = match max_order_result {
            Ok(row) => {
                let max_order: Option<i64> = row.get("max_order");
                max_order.map(|m| m + 1).unwrap_or(0)
            }
            Err(e) => {
                error!("Failed to get max order for conversation {}: {}", id, e);
                return;
            }
        };

        let created_at = chrono::Utc::now().timestamp();

        for (offset, item) in items.into_iter().enumerate() {
            let order = start_order + offset as i64;

            let content_json = match serde_json::to_string(&item) {
                Ok(json) => json,
                Err(e) => {
                    error!("Failed to serialize item: {}", e);
                    continue;
                }
            };

            let result = sqlx::query(
                "INSERT INTO conversation_items (id, conversation_id, item_order, content_json, created_at)
                 VALUES (?, ?, ?, ?, ?)"
            )
            .bind(&item.id)
            .bind(id)
            .bind(order)
            .bind(&content_json)
            .bind(created_at)
            .execute(&self.pool)
            .await;

            if let Err(e) = result {
                error!(
                    "Failed to append item {} to conversation {}: {}",
                    item.id, id, e
                );
            }
        }
    }

    async fn cleanup_expired_async(&self) {
        let now = chrono::Utc::now().timestamp();
        let result = sqlx::query(
            "DELETE FROM conversations WHERE expires_at IS NOT NULL AND expires_at < ?",
        )
        .bind(now)
        .execute(&self.pool)
        .await;

        match result {
            Ok(res) => {
                if res.rows_affected() > 0 {
                    tracing::info!("Cleaned up {} expired conversations", res.rows_affected());
                }
            }
            Err(e) => error!("Failed to cleanup expired conversations: {}", e),
        }
    }

    async fn update_title_async(&self, id: &str, title: &str) -> Result<(), String> {
        // Get current metadata
        let conv = self
            .get_async(id)
            .await
            .ok_or_else(|| format!("Conversation not found: {}", id))?;

        // Update or create metadata with title
        let metadata = conv.metadata.unwrap_or_else(Metadata::new);
        let mut map = metadata.inner().clone();
        map.insert("title".to_string(), title.to_string());
        let new_metadata = Metadata::from(map);

        let metadata_json = serde_json::to_string(&new_metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

        let result = sqlx::query("UPDATE conversations SET metadata_json = ? WHERE id = ?")
            .bind(&metadata_json)
            .bind(id)
            .execute(&self.pool)
            .await;

        match result {
            Ok(res) if res.rows_affected() > 0 => Ok(()),
            Ok(_) => Err(format!("Conversation not found: {}", id)),
            Err(e) => Err(format!("Failed to update title for {}: {}", id, e)),
        }
    }
}

impl ConversationStore for SqliteConversationStore {
    fn list(&self, query: &PaginationQuery) -> ListResponse<Conversation> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let query = query.clone();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.list_async(&query).await })
        })
    }

    fn create(&self, metadata: Option<Metadata>, items: Option<Vec<InputItem>>) -> Conversation {
        let pool = self.pool.clone();
        let store = Self { pool };
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.create_async(metadata, items).await })
        })
    }

    fn get(&self, id: &str) -> Option<Conversation> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.get_async(&id).await })
        })
    }

    fn update(&self, id: &str, metadata: Metadata) -> Option<Conversation> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.update_async(&id, metadata).await })
        })
    }

    fn delete(&self, id: &str) -> bool {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.delete_async(&id).await })
        })
    }

    fn list_items(
        &self,
        id: &str,
        query: &PaginationQuery,
    ) -> Option<ListResponse<ConversationItem>> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        let query = query.clone();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.list_items_async(&id, &query).await })
        })
    }

    fn add_items(&self, id: &str, items: Vec<InputItem>) -> Option<ListResponse<ConversationItem>> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.add_items_async(&id, items).await })
        })
    }

    fn get_item(&self, conv_id: &str, item_id: &str) -> Option<ConversationItem> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let conv_id = conv_id.to_string();
        let item_id = item_id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.get_item_async(&conv_id, &item_id).await })
        })
    }

    fn delete_item(&self, conv_id: &str, item_id: &str) -> Option<Conversation> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let conv_id = conv_id.to_string();
        let item_id = item_id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.delete_item_async(&conv_id, &item_id).await })
        })
    }

    fn get_stored(&self, id: &str) -> Option<StoredConversation> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.get_stored_async(&id).await })
        })
    }

    fn append_output_items(&self, id: &str, items: Vec<ConversationItem>) {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.append_output_items_async(&id, items).await })
        });
    }

    fn cleanup_expired(&self) {
        let pool = self.pool.clone();
        let store = Self { pool };
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.cleanup_expired_async().await })
        });
    }

    fn update_title(&self, id: &str, title: &str) -> Result<(), String> {
        let pool = self.pool.clone();
        let store = Self { pool };
        let id = id.to_string();
        let title = title.to_string();
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async { store.update_title_async(&id, &title).await })
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert an InputItem to a ConversationItem with appropriate ID.
fn input_to_conversation_item(item: InputItem) -> ConversationItem {
    let id = match &item {
        InputItem::Message(_) => message_id(),
        InputItem::Reasoning(_) => reasoning_id(),
        InputItem::FunctionCall(_) => function_call_id(),
        InputItem::CustomToolCall(_) => item_id(),
        InputItem::FunctionCallOutput(_) => item_id(),
        InputItem::CustomToolCallOutput(_) => item_id(),
        InputItem::ComputerCallOutput(_) => item_id(),
        InputItem::ItemReference(r) => r.id.clone(),
    };

    ConversationItem {
        id,
        status: OutputStatus::Completed,
        content: ConversationItemContent::Input(item),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Input, ResponseStatus, ToolChoice, Truncation, Usage};
    use tempfile::TempDir;

    fn make_test_response(id: &str) -> Response {
        Response {
            id: id.to_string(),
            object: Response::OBJECT,
            created_at: 1234567890,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4".to_string(),
            output: vec![],
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: None,
            store: true,
            temperature: 1.0,
            text: None,
            tool_choice: ToolChoice::default(),
            tools: vec![],
            top_p: 1.0,
            truncation: Truncation::default(),
            usage: Usage::default(),
            user: None,
            metadata: serde_json::Value::Null,
        }
    }

    fn make_test_request() -> CreateResponseRequest {
        CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Text("Hello".to_string()),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Truncation::default(),
            metadata: None,
            conversation: None,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_store_and_retrieve() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteResponseStore::new(&db_path).await.unwrap();

        let response = make_test_response("resp_123");
        let request = make_test_request();

        store.store(response.clone(), request);

        let retrieved = store.get("resp_123").unwrap();
        assert_eq!(retrieved.response.id, "resp_123");
        assert_eq!(retrieved.response.model, "gpt-4");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_delete() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteResponseStore::new(&db_path).await.unwrap();

        let response = make_test_response("resp_456");
        let request = make_test_request();

        store.store(response, request);
        assert!(store.delete("resp_456"));
        assert!(store.get("resp_456").is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_list_ids() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteResponseStore::new(&db_path).await.unwrap();

        let request = make_test_request();
        store.store(make_test_response("resp_a"), request.clone());
        store.store(make_test_response("resp_b"), request.clone());
        store.store(make_test_response("resp_c"), request);

        let ids = store.list_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"resp_a".to_string()));
        assert!(ids.contains(&"resp_b".to_string()));
        assert!(ids.contains(&"resp_c".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let store = SqliteResponseStore::new(&db_path).await.unwrap();

        assert!(store.get("resp_nonexistent").is_none());
    }
}
