// ABOUTME: Storage backend configuration.
// Reads from environment variables to determine which backend to use.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};

use crate::state::{ConversationStore, ResponseStore};

use super::generic::{GenericStore, InMemoryGenericStore};
use super::memory::{InMemoryConversationStore, InMemoryStore};
use super::sqlite::{SqliteConversationStore, SqliteResponseStore};

#[derive(Clone, Debug)]
pub enum StorageBackend {
    Memory,
    Sqlite { path: PathBuf },
}

#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub backend: StorageBackend,
}

impl StorageConfig {
    pub fn from_env() -> Self {
        let backend_str = std::env::var("STORAGE_BACKEND").unwrap_or_else(|_| "memory".to_string());

        let backend = match backend_str.to_lowercase().as_str() {
            "sqlite" => {
                let path = std::env::var("STORAGE_SQLITE_PATH")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("/data/store.db"));
                StorageBackend::Sqlite { path }
            }
            _ => StorageBackend::Memory,
        };

        Self { backend }
    }
}

/// Initialize SQLite connection pool and run migrations.
async fn init_sqlite_pool(path: &Path) -> Result<SqlitePool, sqlx::Error> {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await?;

    Ok(pool)
}

pub async fn create_response_store(config: &StorageConfig) -> Arc<dyn ResponseStore + Send + Sync> {
    match &config.backend {
        StorageBackend::Memory => Arc::new(InMemoryStore::new()),
        StorageBackend::Sqlite { path } => match init_sqlite_pool(path).await {
            Ok(pool) => Arc::new(SqliteResponseStore::from_pool(pool)),
            Err(e) => {
                tracing::error!("Failed to initialize SQLite storage: {}", e);
                tracing::warn!("Falling back to in-memory storage");
                Arc::new(InMemoryStore::new())
            }
        },
    }
}

pub async fn create_conversation_store(
    config: &StorageConfig,
) -> Arc<dyn ConversationStore + Send + Sync> {
    match &config.backend {
        StorageBackend::Memory => Arc::new(InMemoryConversationStore::new()),
        StorageBackend::Sqlite { path } => match init_sqlite_pool(path).await {
            Ok(pool) => Arc::new(SqliteConversationStore::new(pool)),
            Err(e) => {
                tracing::error!("Failed to initialize SQLite storage: {}", e);
                tracing::warn!("Falling back to in-memory storage");
                Arc::new(InMemoryConversationStore::new())
            }
        },
    }
}

pub async fn create_generic_store(config: &StorageConfig) -> Arc<dyn GenericStore + Send + Sync> {
    match &config.backend {
        StorageBackend::Memory => Arc::new(InMemoryGenericStore::new()),
        StorageBackend::Sqlite { path: _ } => {
            // SQLite implementation will be added later
            tracing::warn!("SQLite generic storage not yet implemented, using memory");
            Arc::new(InMemoryGenericStore::new())
        }
    }
}
