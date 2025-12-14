//! Conversation storage abstraction.
//!
//! Defines the trait for conversation storage backends.
//! Implementations are in the storage module.

use crate::models::{
    Conversation, ConversationItem, InputItem, ListResponse, Metadata, PaginationQuery,
};

// ============================================================================
// Trait Definition
// ============================================================================

/// Trait for conversation storage backends.
/// Designed to allow swapping DashMap for database-backed storage later.
pub trait ConversationStore: Send + Sync + 'static {
    /// List all conversations with pagination.
    fn list(&self, query: &PaginationQuery) -> ListResponse<Conversation>;

    /// Create a new conversation with optional metadata and initial items.
    fn create(&self, metadata: Option<Metadata>, items: Option<Vec<InputItem>>) -> Conversation;

    /// Get a conversation by ID.
    fn get(&self, id: &str) -> Option<Conversation>;

    /// Update conversation metadata.
    fn update(&self, id: &str, metadata: Metadata) -> Option<Conversation>;

    /// Delete a conversation. Returns true if it existed.
    fn delete(&self, id: &str) -> bool;

    /// List items in a conversation with cursor-based pagination.
    fn list_items(
        &self,
        id: &str,
        query: &PaginationQuery,
    ) -> Option<ListResponse<ConversationItem>>;

    /// Add items to a conversation. Returns the added items.
    fn add_items(&self, id: &str, items: Vec<InputItem>) -> Option<ListResponse<ConversationItem>>;

    /// Get a single item from a conversation.
    fn get_item(&self, conv_id: &str, item_id: &str) -> Option<ConversationItem>;

    /// Delete an item from a conversation. Returns the updated conversation.
    fn delete_item(&self, conv_id: &str, item_id: &str) -> Option<Conversation>;

    /// Get the full stored conversation (for internal use, e.g., response integration).
    /// This returns the StoredConversation type from the storage module.
    fn get_stored(&self, id: &str) -> Option<crate::storage::memory::StoredConversation>;

    /// Append output items to a conversation (after response completes).
    fn append_output_items(&self, id: &str, items: Vec<ConversationItem>);

    /// Clean up expired conversations.
    fn cleanup_expired(&self);

    /// Update the conversation title.
    fn update_title(&self, id: &str, title: &str) -> Result<(), String>;
}
