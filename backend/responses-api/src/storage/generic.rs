//! Generic storage for arbitrary JSON data.
//!
//! Provides a simple key-value store for MCP servers to persist data.
//! Each collection (e.g., `jobs`, `datasets`) is isolated.
//!
//! # URL Structure
//!
//! ```text
//! POST   /v1/storage/jobs          # Create/update record
//! GET    /v1/storage/jobs          # List records
//! GET    /v1/storage/jobs/{id}     # Get single record
//! DELETE /v1/storage/jobs/{id}     # Delete record
//! ```

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// A stored record with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRecord {
    pub id: String,
    pub data: serde_json::Value,
    pub created_at: i64,
    pub updated_at: i64,
}

impl StoredRecord {
    pub fn new(id: String, data: serde_json::Value) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            id,
            data,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_update(mut self, data: serde_json::Value) -> Self {
        self.data = data;
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        self
    }
}

/// Trait for generic storage backends.
pub trait GenericStore: Send + Sync + 'static {
    /// Save or update a record in the given collection.
    fn save(&self, collection: &str, id: &str, data: &serde_json::Value);

    /// Get a record by collection and ID.
    fn get(&self, collection: &str, id: &str) -> Option<StoredRecord>;

    /// List records in a collection with a limit.
    fn list(&self, collection: &str, limit: usize) -> Vec<StoredRecord>;

    /// Delete a record by collection and ID. Returns true if it existed.
    fn delete(&self, collection: &str, id: &str) -> bool;

    /// Clean up expired records.
    fn cleanup_expired(&self);
}

// ============================================================================
// In-Memory Implementation
// ============================================================================

/// In-memory generic store using nested DashMap.
#[derive(Clone)]
pub struct InMemoryGenericStore {
    // Outer map: collection -> inner map
    // Inner map: id -> StoredRecord
    data: Arc<DashMap<String, DashMap<String, StoredRecord>>>,
    default_ttl: Duration,
}

impl InMemoryGenericStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(DashMap::new()),
            default_ttl: Duration::MAX, // No expiration by default
        }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Get or create the inner map for a collection.
    fn get_collection(
        &self,
        collection: &str,
    ) -> dashmap::mapref::one::Ref<'_, String, DashMap<String, StoredRecord>> {
        if !self.data.contains_key(collection) {
            self.data.insert(collection.to_string(), DashMap::new());
        }
        self.data.get(collection).expect("just inserted")
    }
}

impl Default for InMemoryGenericStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GenericStore for InMemoryGenericStore {
    fn save(&self, collection: &str, id: &str, data: &serde_json::Value) {
        let coll_map = self.get_collection(collection);

        if let Some(mut existing) = coll_map.get_mut(id) {
            // Update existing record
            *existing = existing.clone().with_update(data.clone());
        } else {
            // Create new record
            let record = StoredRecord::new(id.to_string(), data.clone());
            coll_map.insert(id.to_string(), record);
        }
    }

    fn get(&self, collection: &str, id: &str) -> Option<StoredRecord> {
        self.data
            .get(collection)
            .and_then(|coll_map| coll_map.get(id).map(|r| r.clone()))
    }

    fn list(&self, collection: &str, limit: usize) -> Vec<StoredRecord> {
        if let Some(coll_map) = self.data.get(collection) {
            let mut records: Vec<StoredRecord> =
                coll_map.iter().map(|entry| entry.value().clone()).collect();

            // Sort by created_at descending (newest first)
            records.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            // Apply limit
            records.truncate(limit);
            records
        } else {
            vec![]
        }
    }

    fn delete(&self, collection: &str, id: &str) -> bool {
        self.data
            .get(collection)
            .map(|coll_map| coll_map.remove(id).is_some())
            .unwrap_or(false)
    }

    fn cleanup_expired(&self) {
        // In-memory store doesn't support TTL per-record yet
        // Could be extended if needed
    }
}

// ============================================================================
// SQLite Implementation (TODO)
// ============================================================================

// Note: SQLite implementation will be added when storage/sqlite.rs is created.
// It will use rusqlite with the block_in_place pattern like other stores.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_get() {
        let store = InMemoryGenericStore::new();
        let data = serde_json::json!({"status": "running", "progress": 50});

        store.save("jobs", "job_123", &data);

        let record = store.get("jobs", "job_123").unwrap();
        assert_eq!(record.id, "job_123");
        assert_eq!(record.data["status"], "running");
        assert_eq!(record.data["progress"], 50);
    }

    #[test]
    fn update_existing() {
        let store = InMemoryGenericStore::new();
        let data1 = serde_json::json!({"status": "running", "progress": 50});
        let data2 = serde_json::json!({"status": "completed", "progress": 100});

        store.save("jobs", "job_123", &data1);
        let record1 = store.get("jobs", "job_123").unwrap();
        let created_at = record1.created_at;

        // Wait a bit to ensure timestamps differ (1 second to ensure visible difference)
        std::thread::sleep(std::time::Duration::from_secs(1));

        store.save("jobs", "job_123", &data2);
        let record2 = store.get("jobs", "job_123").unwrap();

        assert_eq!(record2.id, "job_123");
        assert_eq!(record2.data["status"], "completed");
        assert_eq!(record2.created_at, created_at); // Created timestamp unchanged
        assert!(record2.updated_at > record1.updated_at); // Updated timestamp changed
    }

    #[test]
    fn list_records() {
        let store = InMemoryGenericStore::new();

        store.save("jobs", "job_1", &serde_json::json!({"n": 1}));
        std::thread::sleep(std::time::Duration::from_secs(1));
        store.save("jobs", "job_2", &serde_json::json!({"n": 2}));
        std::thread::sleep(std::time::Duration::from_secs(1));
        store.save("jobs", "job_3", &serde_json::json!({"n": 3}));

        let records = store.list("jobs", 10);
        assert_eq!(records.len(), 3);

        // Should be sorted newest first
        assert_eq!(records[0].id, "job_3");
        assert_eq!(records[1].id, "job_2");
        assert_eq!(records[2].id, "job_1");
    }

    #[test]
    fn list_with_limit() {
        let store = InMemoryGenericStore::new();

        for i in 0..10 {
            store.save("jobs", &format!("job_{}", i), &serde_json::json!({"n": i}));
        }

        let records = store.list("jobs", 3);
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn collection_isolation() {
        let store = InMemoryGenericStore::new();

        store.save("jobs", "id_1", &serde_json::json!({"value": "a"}));
        store.save("datasets", "id_1", &serde_json::json!({"value": "b"}));

        let record1 = store.get("jobs", "id_1").unwrap();
        let record2 = store.get("datasets", "id_1").unwrap();

        assert_eq!(record1.data["value"], "a");
        assert_eq!(record2.data["value"], "b");
    }

    #[test]
    fn delete_record() {
        let store = InMemoryGenericStore::new();

        store.save("jobs", "job_123", &serde_json::json!({"status": "running"}));
        assert!(store.get("jobs", "job_123").is_some());

        let deleted = store.delete("jobs", "job_123");
        assert!(deleted);
        assert!(store.get("jobs", "job_123").is_none());
    }

    #[test]
    fn delete_nonexistent() {
        let store = InMemoryGenericStore::new();
        let deleted = store.delete("jobs", "job_nonexistent");
        assert!(!deleted);
    }

    #[test]
    fn list_empty_collection() {
        let store = InMemoryGenericStore::new();
        let records = store.list("nonexistent", 10);
        assert_eq!(records.len(), 0);
    }
}
