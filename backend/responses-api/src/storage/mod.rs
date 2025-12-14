// ABOUTME: Storage backend configuration and factory functions.
// Supports in-memory (default) and SQLite backends via environment variables.

pub mod config;
pub mod generic;
pub mod memory;
pub mod sqlite;

pub use config::{
    StorageBackend, StorageConfig, create_conversation_store, create_generic_store,
    create_response_store,
};
pub use generic::{GenericStore, InMemoryGenericStore, StoredRecord};
pub use memory::{InMemoryConversationStore, InMemoryStore};
pub use sqlite::{SqliteConversationStore, SqliteResponseStore};
