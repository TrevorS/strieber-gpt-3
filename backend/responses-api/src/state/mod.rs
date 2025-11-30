//! State management for Responses API.
//!
//! Uses DashMap for concurrent in-memory storage of responses.
//! Designed with a trait abstraction for future SQLite/Redis support.

mod store;

pub use store::{InMemoryStore, ResponseStore, StoredResponse};
