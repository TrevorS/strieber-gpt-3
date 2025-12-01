//! Context resolution for previous_response_id chaining.
//!
//! This module handles resolving response chains when a request includes
//! `previous_response_id`. It recursively fetches stored responses and
//! validates the chain before building conversation context.

use std::collections::HashSet;
use std::fmt;

use crate::models::ResponseStatus;
use crate::state::{ResponseStore, StoredResponse};

/// Maximum chain depth (configurable via resolve_chain parameter)
pub const DEFAULT_MAX_CHAIN_DEPTH: usize = 100;

/// Error types for chain resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainErrorKind {
    /// The referenced response ID was not found in storage
    NotFound,
    /// The referenced response has failed status
    FailedResponse,
    /// Chain exceeds maximum allowed depth
    MaxDepthExceeded,
    /// Circular reference detected in chain
    CircularReference,
}

/// Error that can occur during chain resolution.
#[derive(Debug, Clone)]
pub struct ChainResolutionError {
    pub kind: ChainErrorKind,
    pub message: String,
}

impl fmt::Display for ChainResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ChainResolutionError {}

impl ChainResolutionError {
    pub fn not_found(response_id: &str) -> Self {
        Self {
            kind: ChainErrorKind::NotFound,
            message: format!("previous_response_id '{}' not found", response_id),
        }
    }

    pub fn failed_response(response_id: &str) -> Self {
        Self {
            kind: ChainErrorKind::FailedResponse,
            message: format!("cannot chain from failed response '{}'", response_id),
        }
    }

    pub fn max_depth_exceeded(max_depth: usize) -> Self {
        Self {
            kind: ChainErrorKind::MaxDepthExceeded,
            message: format!("conversation chain exceeds maximum depth of {}", max_depth),
        }
    }

    pub fn circular_reference(response_id: &str) -> Self {
        Self {
            kind: ChainErrorKind::CircularReference,
            message: format!(
                "circular reference detected: response '{}' appears multiple times in chain",
                response_id
            ),
        }
    }
}

/// Resolve a chain of previous responses recursively.
///
/// Returns the chain of stored responses in chronological order (oldest first),
/// ready to be assembled into conversation context.
///
/// # Arguments
///
/// * `store` - The response store to fetch responses from
/// * `previous_response_id` - The ID of the most recent previous response
/// * `max_depth` - Maximum chain length allowed (default: 100)
///
/// # Returns
///
/// * `Ok(Vec<StoredResponse>)` - Chain in chronological order (oldest first)
/// * `Err(ChainResolutionError)` - If resolution fails
///
/// # Example
///
/// If Response A → Response B → Response C (current), and we call with B's ID:
/// Returns `[A, B]` (oldest first)
pub fn resolve_chain(
    store: &dyn ResponseStore,
    previous_response_id: &str,
    max_depth: usize,
) -> Result<Vec<StoredResponse>, ChainResolutionError> {
    let mut chain = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut current_id = Some(previous_response_id.to_string());

    tracing::debug!(
        starting_id = %previous_response_id,
        max_depth,
        "Resolving response chain"
    );

    while let Some(id) = current_id.take() {
        // Check for circular reference
        if seen_ids.contains(&id) {
            return Err(ChainResolutionError::circular_reference(&id));
        }

        // Check max depth
        if chain.len() >= max_depth {
            return Err(ChainResolutionError::max_depth_exceeded(max_depth));
        }

        // Fetch the response
        let stored = store
            .get(&id)
            .ok_or_else(|| ChainResolutionError::not_found(&id))?;

        // Validate response status
        if stored.response.status == ResponseStatus::Failed {
            return Err(ChainResolutionError::failed_response(&id));
        }

        // Track this ID
        seen_ids.insert(id.clone());

        // Check for next link in chain
        current_id = stored.response.previous_response_id.clone();

        // Add to chain (we'll reverse at the end)
        chain.push(stored);
    }

    // Reverse to get chronological order (oldest first)
    chain.reverse();

    tracing::debug!(
        chain_length = chain.len(),
        chain_ids = ?chain.iter().map(|s| &s.response.id).collect::<Vec<_>>(),
        "Chain resolved successfully"
    );

    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CreateResponseRequest, Input, Response, ToolChoice, Truncation, Usage};
    use crate::state::InMemoryStore;

    fn make_test_response(id: &str, prev_id: Option<&str>, status: ResponseStatus) -> Response {
        Response {
            id: id.to_string(),
            object: Response::OBJECT,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4".to_string(),
            output: vec![],
            parallel_tool_calls: true,
            previous_response_id: prev_id.map(String::from),
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
    fn single_response_chain() {
        let store = InMemoryStore::new();
        let resp = make_test_response("resp_a", None, ResponseStatus::Completed);
        let req = make_test_request();
        store.store(resp, req);

        let chain = resolve_chain(&store, "resp_a", 100).expect("should resolve");

        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].response.id, "resp_a");
    }

    #[test]
    fn two_response_chain_in_chronological_order() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // A is older, B points to A
        let resp_a = make_test_response("resp_a", None, ResponseStatus::Completed);
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Completed);

        store.store(resp_a, req.clone());
        store.store(resp_b, req);

        let chain = resolve_chain(&store, "resp_b", 100).expect("should resolve");

        assert_eq!(chain.len(), 2);
        // Chronological order: A first (older), then B
        assert_eq!(chain[0].response.id, "resp_a");
        assert_eq!(chain[1].response.id, "resp_b");
    }

    #[test]
    fn three_response_chain_in_chronological_order() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // A → B → C (C is newest)
        let resp_a = make_test_response("resp_a", None, ResponseStatus::Completed);
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Completed);
        let resp_c = make_test_response("resp_c", Some("resp_b"), ResponseStatus::Completed);

        store.store(resp_a, req.clone());
        store.store(resp_b, req.clone());
        store.store(resp_c, req);

        let chain = resolve_chain(&store, "resp_c", 100).expect("should resolve");

        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].response.id, "resp_a");
        assert_eq!(chain[1].response.id, "resp_b");
        assert_eq!(chain[2].response.id, "resp_c");
    }

    #[test]
    fn not_found_error() {
        let store = InMemoryStore::new();

        let err = resolve_chain(&store, "nonexistent", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::NotFound);
        assert!(err.message.contains("nonexistent"));
    }

    #[test]
    fn failed_response_error() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        let resp = make_test_response("resp_failed", None, ResponseStatus::Failed);
        store.store(resp, req);

        let err = resolve_chain(&store, "resp_failed", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::FailedResponse);
        assert!(err.message.contains("resp_failed"));
    }

    #[test]
    fn failed_response_in_middle_of_chain_error() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // A is OK, B failed, C points to B
        let resp_a = make_test_response("resp_a", None, ResponseStatus::Completed);
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Failed);
        let resp_c = make_test_response("resp_c", Some("resp_b"), ResponseStatus::Completed);

        store.store(resp_a, req.clone());
        store.store(resp_b, req.clone());
        store.store(resp_c, req);

        // Resolving from C should fail when it hits B
        let err = resolve_chain(&store, "resp_c", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::FailedResponse);
        assert!(err.message.contains("resp_b"));
    }

    #[test]
    fn max_depth_exceeded_error() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // Create a chain of 5 responses
        let resp_a = make_test_response("resp_a", None, ResponseStatus::Completed);
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Completed);
        let resp_c = make_test_response("resp_c", Some("resp_b"), ResponseStatus::Completed);
        let resp_d = make_test_response("resp_d", Some("resp_c"), ResponseStatus::Completed);
        let resp_e = make_test_response("resp_e", Some("resp_d"), ResponseStatus::Completed);

        store.store(resp_a, req.clone());
        store.store(resp_b, req.clone());
        store.store(resp_c, req.clone());
        store.store(resp_d, req.clone());
        store.store(resp_e, req);

        // Max depth of 3 should fail
        let err = resolve_chain(&store, "resp_e", 3).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::MaxDepthExceeded);
        assert!(err.message.contains("3"));
    }

    #[test]
    fn circular_reference_detected() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // This is an edge case - we need to manually create a circular reference
        // In practice this shouldn't happen, but we should handle it
        // A → B → A (circular)
        let resp_a = make_test_response("resp_a", Some("resp_b"), ResponseStatus::Completed);
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Completed);

        store.store(resp_a, req.clone());
        store.store(resp_b, req);

        let err = resolve_chain(&store, "resp_a", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::CircularReference);
    }

    #[test]
    fn self_referential_detected() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // Response pointing to itself
        let resp = make_test_response("resp_a", Some("resp_a"), ResponseStatus::Completed);
        store.store(resp, req);

        let err = resolve_chain(&store, "resp_a", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::CircularReference);
    }

    #[test]
    fn broken_chain_returns_not_found() {
        let store = InMemoryStore::new();
        let req = make_test_request();

        // B points to A, but A doesn't exist
        let resp_b = make_test_response("resp_b", Some("resp_a"), ResponseStatus::Completed);
        store.store(resp_b, req);

        let err = resolve_chain(&store, "resp_b", 100).expect_err("should fail");

        assert_eq!(err.kind, ChainErrorKind::NotFound);
        assert!(err.message.contains("resp_a"));
    }
}
