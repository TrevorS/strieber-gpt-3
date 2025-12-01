//! Container file storage with DashMap implementation.
//!
//! Containers are simple file storage buckets for code interpreter outputs.
//! They don't maintain Python state - just store generated files (images, etc.)
//! that can be downloaded via the API.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

/// Default TTL for containers (20 minutes of inactivity).
pub const DEFAULT_CONTAINER_TTL_SECS: u64 = 1200;

/// Generate a unique container ID.
pub fn container_id() -> String {
    format!("cntr_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a unique file ID within a container.
pub fn container_file_id() -> String {
    format!("cfile_{}", uuid::Uuid::new_v4().simple())
}

/// A file stored within a container.
#[derive(Debug, Clone)]
pub struct ContainerFile {
    /// Unique file identifier.
    pub id: String,
    /// Original filename (e.g., "output_0.png").
    pub filename: String,
    /// File content (decoded from base64).
    pub content: Vec<u8>,
    /// MIME type (e.g., "image/png").
    pub mime_type: String,
    /// When the file was created.
    pub created_at: i64,
}

/// Internal state for a container.
#[derive(Debug, Clone)]
pub struct ContainerState {
    /// Container ID.
    pub id: String,
    /// Files stored in this container.
    pub files: HashMap<String, ContainerFile>,
    /// When the container was created.
    pub created_at: i64,
    /// Last time the container was accessed.
    pub last_used: Instant,
    /// Time-to-live after last use.
    pub ttl: Duration,
}

impl ContainerState {
    /// Create a new container state.
    pub fn new(id: String) -> Self {
        Self {
            id,
            files: HashMap::new(),
            created_at: unix_timestamp(),
            last_used: Instant::now(),
            ttl: Duration::from_secs(DEFAULT_CONTAINER_TTL_SECS),
        }
    }

    /// Update the last-used timestamp.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    /// Check if the container has expired.
    pub fn is_expired(&self) -> bool {
        self.last_used.elapsed() > self.ttl
    }
}

/// Container store using DashMap for concurrent access.
#[derive(Clone)]
pub struct ContainerStore {
    containers: Arc<DashMap<String, ContainerState>>,
}

impl Default for ContainerStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainerStore {
    /// Create a new container store.
    pub fn new() -> Self {
        Self {
            containers: Arc::new(DashMap::new()),
        }
    }

    /// Create a new container and return its ID.
    pub fn create(&self) -> String {
        let id = container_id();
        let state = ContainerState::new(id.clone());
        self.containers.insert(id.clone(), state);
        tracing::debug!("Created container: {}", id);
        id
    }

    /// Get or create a container, returning its ID.
    /// If a container_id is provided and exists, returns it. Otherwise creates a new one.
    pub fn get_or_create(&self, container_id: Option<&str>) -> String {
        if let Some(id) = container_id {
            if self.containers.contains_key(id) {
                // Touch to update last_used
                if let Some(mut entry) = self.containers.get_mut(id) {
                    entry.touch();
                }
                return id.to_string();
            }
        }
        self.create()
    }

    /// Add a file to a container.
    /// Returns the file ID if successful.
    pub fn add_file(
        &self,
        container_id: &str,
        filename: String,
        content: Vec<u8>,
        mime_type: &str,
    ) -> Option<String> {
        let mut entry = self.containers.get_mut(container_id)?;
        entry.touch();

        let file_id = container_file_id();
        let file = ContainerFile {
            id: file_id.clone(),
            filename: filename.clone(),
            content,
            mime_type: mime_type.to_string(),
            created_at: unix_timestamp(),
        };

        tracing::debug!(
            "Added file {} ({}) to container {}",
            file_id,
            filename,
            container_id
        );
        entry.files.insert(file_id.clone(), file);
        Some(file_id)
    }

    /// Get a file's content and MIME type.
    pub fn get_file_content(
        &self,
        container_id: &str,
        file_id: &str,
    ) -> Option<(Vec<u8>, String)> {
        let entry = self.containers.get(container_id)?;
        let file = entry.files.get(file_id)?;
        Some((file.content.clone(), file.mime_type.clone()))
    }

    /// Get file metadata without content.
    pub fn get_file_metadata(
        &self,
        container_id: &str,
        file_id: &str,
    ) -> Option<(String, String, usize)> {
        let entry = self.containers.get(container_id)?;
        let file = entry.files.get(file_id)?;
        Some((file.filename.clone(), file.mime_type.clone(), file.content.len()))
    }

    /// List all file IDs in a container.
    pub fn list_files(&self, container_id: &str) -> Option<Vec<String>> {
        let entry = self.containers.get(container_id)?;
        Some(entry.files.keys().cloned().collect())
    }

    /// Check if a container exists.
    pub fn exists(&self, container_id: &str) -> bool {
        self.containers.contains_key(container_id)
    }

    /// Clean up expired containers.
    pub fn cleanup_expired(&self) {
        let before = self.containers.len();
        self.containers.retain(|_, state| !state.is_expired());
        let after = self.containers.len();
        if before != after {
            tracing::debug!("Cleaned up {} expired containers", before - after);
        }
    }

    /// Get count of containers (for debugging).
    pub fn len(&self) -> usize {
        self.containers.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }
}

/// Get current Unix timestamp.
fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time went backwards")
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_container() {
        let store = ContainerStore::new();
        let id = store.create();
        assert!(id.starts_with("cntr_"));
        assert!(store.exists(&id));
    }

    #[test]
    fn add_and_get_file() {
        let store = ContainerStore::new();
        let container_id = store.create();

        let content = b"PNG image data".to_vec();
        let file_id = store
            .add_file(&container_id, "test.png".to_string(), content.clone(), "image/png")
            .unwrap();

        assert!(file_id.starts_with("cfile_"));

        let (retrieved_content, mime) = store.get_file_content(&container_id, &file_id).unwrap();
        assert_eq!(retrieved_content, content);
        assert_eq!(mime, "image/png");
    }

    #[test]
    fn get_or_create_existing() {
        let store = ContainerStore::new();
        let id1 = store.create();
        let id2 = store.get_or_create(Some(&id1));
        assert_eq!(id1, id2);
    }

    #[test]
    fn get_or_create_new() {
        let store = ContainerStore::new();
        let id = store.get_or_create(None);
        assert!(id.starts_with("cntr_"));
        assert!(store.exists(&id));
    }

    #[test]
    fn nonexistent_container_returns_none() {
        let store = ContainerStore::new();
        assert!(store.get_file_content("cntr_fake", "file_fake").is_none());
        assert!(store.add_file("cntr_fake", "test.png".to_string(), vec![], "image/png").is_none());
    }

    #[test]
    fn cleanup_expired_containers() {
        let store = ContainerStore::new();
        let id = store.create();

        // Set TTL to 0 (immediately expired)
        if let Some(mut entry) = store.containers.get_mut(&id) {
            entry.ttl = Duration::ZERO;
        }

        // Wait briefly and cleanup
        std::thread::sleep(Duration::from_millis(1));
        store.cleanup_expired();

        assert!(!store.exists(&id));
    }
}
