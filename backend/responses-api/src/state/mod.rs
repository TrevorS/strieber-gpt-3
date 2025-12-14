//! State management for Responses API.
//!
//! Defines storage traits and re-exports implementations from the storage module.

mod conversation_store;
mod store;

pub use conversation_store::ConversationStore;
pub use store::{ResponseStore, StoredResponse};

// Re-export implementations from storage module for backward compatibility
pub use crate::storage::{InMemoryConversationStore, InMemoryStore};
