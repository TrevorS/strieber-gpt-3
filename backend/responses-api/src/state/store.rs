//! Response storage abstraction with DashMap implementation.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::models::{CreateResponseRequest, Response};

/// Stored response with metadata.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    /// The response object
    pub response: Response,
    /// Original request (for potential replay)
    pub request: CreateResponseRequest,
    /// When this was created
    pub created_at: Instant,
    /// Time-to-live (after which it can be cleaned up)
    pub ttl: Duration,
}

impl StoredResponse {
    pub fn new(response: Response, request: CreateResponseRequest) -> Self {
        Self {
            response,
            request,
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

/// Trait for response storage backends.
pub trait ResponseStore: Send + Sync + 'static {
    /// Store a response.
    fn store(&self, response: Response, request: CreateResponseRequest);

    /// Get a response by ID.
    fn get(&self, id: &str) -> Option<StoredResponse>;

    /// Delete a response by ID. Returns true if it existed.
    fn delete(&self, id: &str) -> bool;

    /// List all response IDs (for debugging/admin).
    fn list_ids(&self) -> Vec<String>;

    /// Clean up expired responses.
    fn cleanup_expired(&self);
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Input, ResponseStatus, ToolChoice, Truncation, Usage};

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
}
