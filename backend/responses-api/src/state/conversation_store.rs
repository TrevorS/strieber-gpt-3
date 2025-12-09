//! Conversation storage abstraction with DashMap implementation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::models::{
    Conversation, ConversationItem, ConversationItemContent, InputItem, ListResponse, Metadata,
    OutputStatus, PaginationQuery, SortOrder,
};
use crate::translation::{conversation_id, function_call_id, item_id, message_id, reasoning_id};

// ============================================================================
// Stored Types
// ============================================================================

/// A stored conversation with its items and metadata.
#[derive(Debug, Clone)]
pub struct StoredConversation {
    pub conversation: Conversation,
    pub items: Vec<ConversationItem>,
    pub created_at: Instant,
    pub ttl: Duration,
}

impl StoredConversation {
    pub fn new(conversation: Conversation, items: Vec<ConversationItem>) -> Self {
        Self {
            conversation,
            items,
            created_at: Instant::now(),
            ttl: Duration::from_secs(3600), // 1 hour default
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

// ============================================================================
// Trait Definition
// ============================================================================

/// Trait for conversation storage backends.
/// Designed to allow swapping DashMap for database-backed storage later.
pub trait ConversationStore: Send + Sync + 'static {
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
    fn get_stored(&self, id: &str) -> Option<StoredConversation>;

    /// Append output items to a conversation (after response completes).
    fn append_output_items(&self, id: &str, items: Vec<ConversationItem>);

    /// Clean up expired conversations.
    fn cleanup_expired(&self);
}

// ============================================================================
// In-Memory Implementation
// ============================================================================

/// In-memory conversation store using DashMap for concurrent access.
#[derive(Clone)]
pub struct InMemoryConversationStore {
    conversations: Arc<DashMap<String, StoredConversation>>,
    default_ttl: Duration,
}

impl InMemoryConversationStore {
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(DashMap::new()),
            default_ttl: Duration::from_secs(3600),
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    pub fn len(&self) -> usize {
        self.conversations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conversations.is_empty()
    }
}

impl Default for InMemoryConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationStore for InMemoryConversationStore {
    fn create(&self, metadata: Option<Metadata>, items: Option<Vec<InputItem>>) -> Conversation {
        let id = conversation_id();
        let conversation = Conversation::new(id.clone(), metadata);

        // Convert input items to conversation items
        let conv_items: Vec<ConversationItem> = items
            .unwrap_or_default()
            .into_iter()
            .map(input_to_conversation_item)
            .collect();

        let stored =
            StoredConversation::new(conversation.clone(), conv_items).with_ttl(self.default_ttl);

        self.conversations.insert(id, stored);
        conversation
    }

    fn get(&self, id: &str) -> Option<Conversation> {
        self.conversations.get(id).map(|s| s.conversation.clone())
    }

    fn update(&self, id: &str, metadata: Metadata) -> Option<Conversation> {
        self.conversations.get_mut(id).map(|mut stored| {
            stored.conversation.metadata = Some(metadata);
            stored.conversation.clone()
        })
    }

    fn delete(&self, id: &str) -> bool {
        self.conversations.remove(id).is_some()
    }

    fn list_items(
        &self,
        id: &str,
        query: &PaginationQuery,
    ) -> Option<ListResponse<ConversationItem>> {
        let stored = self.conversations.get(id)?;
        let items = &stored.items;

        if items.is_empty() {
            return Some(ListResponse::empty());
        }

        // Find start position based on cursor
        let start_idx = if let Some(ref after) = query.after {
            items
                .iter()
                .position(|i| i.id == *after)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };

        // Apply ordering and limit
        let limit = query.limit as usize;
        let (data, has_more) = match query.order {
            SortOrder::Asc => {
                let end_idx = (start_idx + limit).min(items.len());
                let slice = &items[start_idx..end_idx];
                (slice.to_vec(), end_idx < items.len())
            }
            SortOrder::Desc => {
                // For desc, we iterate from end to start
                let reversed: Vec<_> = items.iter().rev().collect();
                let start = if let Some(ref after) = query.after {
                    reversed
                        .iter()
                        .position(|i| i.id == *after)
                        .map(|p| p + 1)
                        .unwrap_or(0)
                } else {
                    0
                };
                let end = (start + limit).min(reversed.len());
                let slice: Vec<_> = reversed[start..end].iter().map(|i| (*i).clone()).collect();
                (slice, end < reversed.len())
            }
        };

        let first_id = data.first().map(|i| i.id.clone());
        let last_id = data.last().map(|i| i.id.clone());

        Some(ListResponse::new(data, has_more).with_ids(first_id, last_id))
    }

    fn add_items(&self, id: &str, items: Vec<InputItem>) -> Option<ListResponse<ConversationItem>> {
        let mut stored = self.conversations.get_mut(id)?;

        let new_items: Vec<ConversationItem> = items
            .into_iter()
            .map(input_to_conversation_item)
            .collect();

        let first_id = new_items.first().map(|i| i.id.clone());
        let last_id = new_items.last().map(|i| i.id.clone());

        stored.items.extend(new_items.clone());

        Some(ListResponse::new(new_items, false).with_ids(first_id, last_id))
    }

    fn get_item(&self, conv_id: &str, item_id: &str) -> Option<ConversationItem> {
        let stored = self.conversations.get(conv_id)?;
        stored.items.iter().find(|i| i.id == item_id).cloned()
    }

    fn delete_item(&self, conv_id: &str, item_id: &str) -> Option<Conversation> {
        let mut stored = self.conversations.get_mut(conv_id)?;
        stored.items.retain(|i| i.id != item_id);
        Some(stored.conversation.clone())
    }

    fn get_stored(&self, id: &str) -> Option<StoredConversation> {
        self.conversations.get(id).map(|s| s.clone())
    }

    fn append_output_items(&self, id: &str, items: Vec<ConversationItem>) {
        if let Some(mut stored) = self.conversations.get_mut(id) {
            stored.items.extend(items);
        }
    }

    fn cleanup_expired(&self) {
        self.conversations.retain(|_, v| !v.is_expired());
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
        InputItem::FunctionCallOutput(_) => item_id(),
        InputItem::CustomToolCallOutput(_) => item_id(),
        InputItem::ComputerCallOutput(_) => item_id(),
    };

    ConversationItem {
        id,
        status: OutputStatus::Completed,
        content: ConversationItemContent::Input(item),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MessageContent, MessageInput, Role};

    fn make_message_item(text: &str) -> InputItem {
        InputItem::Message(MessageInput {
            role: Role::User,
            content: MessageContent::Text(text.to_string()),
        })
    }

    #[test]
    fn create_conversation_generates_id() {
        let store = InMemoryConversationStore::new();
        let conv = store.create(None, None);
        assert!(conv.id.starts_with("conv_"));
        assert_eq!(conv.object, "conversation");
    }

    #[test]
    fn create_with_items() {
        let store = InMemoryConversationStore::new();
        let items = vec![make_message_item("Hello"), make_message_item("World")];
        let conv = store.create(None, Some(items));

        let list = store
            .list_items(&conv.id, &PaginationQuery::default())
            .unwrap();
        assert_eq!(list.data.len(), 2);
    }

    #[test]
    fn get_returns_none_for_missing() {
        let store = InMemoryConversationStore::new();
        assert!(store.get("conv_nonexistent").is_none());
    }

    #[test]
    fn update_metadata() {
        let store = InMemoryConversationStore::new();
        let conv = store.create(None, None);

        let mut meta = std::collections::HashMap::new();
        meta.insert("key".to_string(), "value".to_string());
        let metadata = Metadata::from(meta);

        let updated = store.update(&conv.id, metadata).unwrap();
        assert!(updated.metadata.is_some());
    }

    #[test]
    fn delete_conversation() {
        let store = InMemoryConversationStore::new();
        let conv = store.create(None, None);

        assert!(store.delete(&conv.id));
        assert!(store.get(&conv.id).is_none());
    }

    #[test]
    fn add_and_list_items() {
        let store = InMemoryConversationStore::new();
        let conv = store.create(None, None);

        let items = vec![make_message_item("Test message")];
        let added = store.add_items(&conv.id, items).unwrap();

        assert_eq!(added.data.len(), 1);
        assert!(added.data[0].id.starts_with("msg_"));
    }

    #[test]
    fn pagination_with_limit() {
        let store = InMemoryConversationStore::new();
        let items: Vec<InputItem> = (0..10)
            .map(|i| make_message_item(&format!("msg {}", i)))
            .collect();
        let conv = store.create(None, Some(items));

        let query = PaginationQuery {
            after: None,
            limit: 3,
            order: SortOrder::Asc,
        };
        let list = store.list_items(&conv.id, &query).unwrap();

        assert_eq!(list.data.len(), 3);
        assert!(list.has_more);
    }

    #[test]
    fn get_single_item() {
        let store = InMemoryConversationStore::new();
        let items = vec![make_message_item("Hello")];
        let conv = store.create(None, Some(items));

        let list = store
            .list_items(&conv.id, &PaginationQuery::default())
            .unwrap();
        let item_id = &list.data[0].id;

        let item = store.get_item(&conv.id, item_id).unwrap();
        assert_eq!(item.id, *item_id);
    }

    #[test]
    fn delete_item() {
        let store = InMemoryConversationStore::new();
        let items = vec![make_message_item("Hello"), make_message_item("World")];
        let conv = store.create(None, Some(items));

        let list = store
            .list_items(&conv.id, &PaginationQuery::default())
            .unwrap();
        let item_id = list.data[0].id.clone();

        store.delete_item(&conv.id, &item_id);

        let updated_list = store
            .list_items(&conv.id, &PaginationQuery::default())
            .unwrap();
        assert_eq!(updated_list.data.len(), 1);
    }

    #[test]
    fn expired_conversation_is_cleaned() {
        let store = InMemoryConversationStore::new().with_ttl(Duration::ZERO);
        let conv = store.create(None, None);

        std::thread::sleep(Duration::from_millis(1));
        store.cleanup_expired();

        assert!(store.get(&conv.id).is_none());
    }
}
