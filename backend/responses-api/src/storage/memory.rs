//! In-memory storage implementations using DashMap.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::models::{
    Conversation, ConversationItem, ConversationItemContent, CreateResponseRequest, InputItem,
    ListResponse, Metadata, OutputStatus, PaginationQuery, Response, SortOrder,
};
use crate::state::{ConversationStore, ResponseStore, StoredResponse};
use crate::translation::{conversation_id, function_call_id, item_id, message_id, reasoning_id};

// ============================================================================
// Response Store Implementation
// ============================================================================

/// In-memory response store using DashMap.
#[derive(Clone)]
pub struct InMemoryStore {
    responses: Arc<DashMap<String, StoredResponse>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(DashMap::new()),
        }
    }

    /// Get the number of stored responses.
    pub fn len(&self) -> usize {
        self.responses.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.responses.is_empty()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseStore for InMemoryStore {
    fn store(&self, response: Response, request: CreateResponseRequest) {
        let id = response.id.clone();
        let stored = StoredResponse::new(response, request);
        self.responses.insert(id, stored);
    }

    fn get(&self, id: &str) -> Option<StoredResponse> {
        self.responses.get(id).map(|r| r.value().clone())
    }

    fn delete(&self, id: &str) -> bool {
        self.responses.remove(id).is_some()
    }

    fn list_ids(&self) -> Vec<String> {
        self.responses.iter().map(|r| r.key().clone()).collect()
    }

    fn cleanup_expired(&self) {
        self.responses.retain(|_, v| !v.is_expired());
    }
}

// ============================================================================
// Conversation Store Implementation
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
            ttl: Duration::MAX, // Infinite by default (no expiration)
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
            default_ttl: Duration::MAX, // Infinite by default
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
    fn list(&self, query: &PaginationQuery) -> ListResponse<Conversation> {
        // Collect all conversations, sorted by created_at
        let mut conversations: Vec<Conversation> = self
            .conversations
            .iter()
            .map(|entry| entry.value().conversation.clone())
            .collect();

        // Sort by created_at (desc by default, asc if requested)
        match query.order {
            SortOrder::Desc => conversations.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
            SortOrder::Asc => conversations.sort_by(|a, b| a.created_at.cmp(&b.created_at)),
        }

        if conversations.is_empty() {
            return ListResponse::empty();
        }

        // Find start position based on cursor
        let start_idx = if let Some(ref after) = query.after {
            conversations
                .iter()
                .position(|c| c.id == *after)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };

        // Apply limit
        let limit = query.limit as usize;
        let end_idx = (start_idx + limit).min(conversations.len());
        let data: Vec<Conversation> = conversations[start_idx..end_idx].to_vec();
        let has_more = end_idx < conversations.len();

        let first_id = data.first().map(|c| c.id.clone());
        let last_id = data.last().map(|c| c.id.clone());

        ListResponse::new(data, has_more).with_ids(first_id, last_id)
    }

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

        let new_items: Vec<ConversationItem> =
            items.into_iter().map(input_to_conversation_item).collect();

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

    fn update_title(&self, id: &str, title: &str) -> Result<(), String> {
        if let Some(mut stored) = self.conversations.get_mut(id) {
            let metadata = stored
                .conversation
                .metadata
                .get_or_insert_with(Metadata::new);
            // Get the inner HashMap and insert the title
            // We need to replace the whole metadata since Metadata is a newtype
            let mut map = metadata.inner().clone();
            map.insert("title".to_string(), title.to_string());
            stored.conversation.metadata = Some(Metadata::from(map));
            Ok(())
        } else {
            Err(format!("Conversation not found: {}", id))
        }
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
        InputItem::CustomToolCall(_) => item_id(), // Custom tool calls get item_xxx IDs
        InputItem::FunctionCallOutput(_) => item_id(),
        InputItem::CustomToolCallOutput(_) => item_id(),
        InputItem::ComputerCallOutput(_) => item_id(),
        InputItem::ItemReference(r) => r.id.clone(), // Use the referenced ID
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
    use crate::models::{
        Input, MessageContent, MessageInput, ResponseStatus, Role, ToolChoice, Truncation, Usage,
    };

    // ========================================================================
    // Response Store Tests
    // ========================================================================

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

    #[test]
    fn store_and_retrieve_response() {
        let store = InMemoryStore::new();
        let response = make_test_response("resp_123");
        let request = make_test_request();

        store.store(response.clone(), request);

        let retrieved = store.get("resp_123").unwrap();
        assert_eq!(retrieved.response.id, "resp_123");
    }

    #[test]
    fn delete_response() {
        let store = InMemoryStore::new();
        let response = make_test_response("resp_456");
        let request = make_test_request();

        store.store(response, request);
        assert!(store.delete("resp_456"));
        assert!(store.get("resp_456").is_none());
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let store = InMemoryStore::new();
        assert!(store.get("resp_nonexistent").is_none());
    }

    #[test]
    fn list_ids_returns_all_stored() {
        let store = InMemoryStore::new();
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

    #[test]
    fn expired_response_is_cleaned() {
        let store = InMemoryStore::new();
        let response = make_test_response("resp_expired");
        let request = make_test_request();

        // Store with 0 TTL (immediately expired)
        let id = response.id.clone();
        let stored = StoredResponse::new(response, request).with_ttl(Duration::ZERO);
        store.responses.insert(id, stored);

        // Should be retrievable before cleanup
        assert!(store.get("resp_expired").is_some());

        // Wait a tiny bit and cleanup
        std::thread::sleep(Duration::from_millis(1));
        store.cleanup_expired();

        // Should be gone after cleanup
        assert!(store.get("resp_expired").is_none());
    }

    // ========================================================================
    // Conversation Store Tests
    // ========================================================================

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
