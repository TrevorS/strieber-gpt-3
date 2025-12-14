//! Response storage abstraction.
//!
//! Defines the trait for response storage backends and the StoredResponse type.
//! Implementations are in the storage module.

use std::time::{Duration, Instant};

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
