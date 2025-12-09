//! Common types shared across API models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::{Validate, ValidationError};

// ============================================================================
// Metadata
// ============================================================================

/// Validated metadata map.
/// Constraints: max 16 key-value pairs, keys ≤64 chars, values ≤512 chars.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Metadata(HashMap<String, String>);

impl Metadata {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn into_inner(self) -> HashMap<String, String> {
        self.0
    }

    pub fn inner(&self) -> &HashMap<String, String> {
        &self.0
    }

    /// Validate metadata constraints.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.0.len() > 16 {
            let mut err = ValidationError::new("metadata_max_pairs");
            err.message = Some("Metadata cannot have more than 16 key-value pairs".into());
            return Err(err);
        }
        for (key, value) in &self.0 {
            if key.len() > 64 {
                let mut err = ValidationError::new("metadata_key_length");
                err.message = Some(format!("Metadata key '{}' exceeds 64 characters", key).into());
                return Err(err);
            }
            if value.len() > 512 {
                let mut err = ValidationError::new("metadata_value_length");
                err.message =
                    Some(format!("Metadata value for key '{}' exceeds 512 characters", key).into());
                return Err(err);
            }
        }
        Ok(())
    }
}

impl From<HashMap<String, String>> for Metadata {
    fn from(map: HashMap<String, String>) -> Self {
        Self(map)
    }
}

// ============================================================================
// Pagination
// ============================================================================

/// Pagination query parameters for list endpoints.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct PaginationQuery {
    /// Cursor: return items after this ID
    pub after: Option<String>,

    /// Number of items to return (1-100, default 20)
    #[validate(range(min = 1, max = 100))]
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Sort order (asc or desc, default desc)
    #[serde(default)]
    pub order: SortOrder,
}

impl Default for PaginationQuery {
    fn default() -> Self {
        Self {
            after: None,
            limit: default_limit(),
            order: SortOrder::default(),
        }
    }
}

fn default_limit() -> u32 {
    20
}

// ============================================================================
// Items Query (Pagination + Include)
// ============================================================================

/// Query parameters for items list endpoints.
/// Combines pagination with the include parameter for expanding nested content.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct ItemsListQuery {
    /// Cursor: return items after this ID
    pub after: Option<String>,

    /// Number of items to return (1-100, default 20)
    #[validate(range(min = 1, max = 100))]
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Sort order (asc or desc, default desc)
    #[serde(default)]
    pub order: SortOrder,

    /// Additional data to include in the response.
    /// Valid values include:
    /// - `message.input_image.image_url`
    /// - `message.output_text.logprobs`
    /// - `computer_call_output.output.image_url`
    /// - `reasoning.encrypted_content`
    #[serde(default)]
    pub include: Option<Vec<String>>,
}

impl Default for ItemsListQuery {
    fn default() -> Self {
        Self {
            after: None,
            limit: default_limit(),
            order: SortOrder::default(),
            include: None,
        }
    }
}

impl ItemsListQuery {
    /// Convert to a PaginationQuery for storage layer operations.
    pub fn to_pagination(&self) -> PaginationQuery {
        PaginationQuery {
            after: self.after.clone(),
            limit: self.limit,
            order: self.order,
        }
    }

    /// Check if a specific include value is requested.
    pub fn includes(&self, value: &str) -> bool {
        self.include
            .as_ref()
            .map(|v| v.iter().any(|s| s == value))
            .unwrap_or(false)
    }

    /// Check if any message-related includes are requested.
    pub fn includes_message_data(&self) -> bool {
        self.includes("message.input_image.image_url")
            || self.includes("message.output_text.logprobs")
    }

    /// Check if computer call output image URL is requested.
    pub fn includes_computer_call_image(&self) -> bool {
        self.includes("computer_call_output.output.image_url")
    }

    /// Check if reasoning encrypted content is requested.
    pub fn includes_reasoning_encrypted(&self) -> bool {
        self.includes("reasoning.encrypted_content")
    }
}

/// Sort order for pagination.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Asc,
    #[default]
    Desc,
}

// ============================================================================
// List Response
// ============================================================================

/// Generic paginated list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse<T> {
    pub object: &'static str,
    pub data: Vec<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    pub has_more: bool,
}

impl<T> ListResponse<T> {
    pub const OBJECT: &'static str = "list";

    pub fn new(data: Vec<T>, has_more: bool) -> Self {
        let first_id = None;
        let last_id = None;
        Self {
            object: Self::OBJECT,
            data,
            first_id,
            last_id,
            has_more,
        }
    }

    pub fn with_ids(mut self, first_id: Option<String>, last_id: Option<String>) -> Self {
        self.first_id = first_id;
        self.last_id = last_id;
        self
    }

    pub fn empty() -> Self {
        Self {
            object: Self::OBJECT,
            data: vec![],
            first_id: None,
            last_id: None,
            has_more: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_accepts_valid_data() {
        let mut map = HashMap::new();
        map.insert("key1".to_string(), "value1".to_string());
        map.insert("key2".to_string(), "value2".to_string());
        let metadata = Metadata::from(map);
        assert!(metadata.validate().is_ok());
    }

    #[test]
    fn metadata_rejects_too_many_pairs() {
        let mut map = HashMap::new();
        for i in 0..17 {
            map.insert(format!("key{}", i), format!("value{}", i));
        }
        let metadata = Metadata::from(map);
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn metadata_rejects_long_key() {
        let mut map = HashMap::new();
        map.insert("k".repeat(65), "value".to_string());
        let metadata = Metadata::from(map);
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn metadata_rejects_long_value() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), "v".repeat(513));
        let metadata = Metadata::from(map);
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn pagination_default_values() {
        let query = PaginationQuery::default();
        assert_eq!(query.limit, 20);
        assert_eq!(query.order, SortOrder::Desc);
        assert!(query.after.is_none());
    }

    #[test]
    fn list_response_empty() {
        let response: ListResponse<String> = ListResponse::empty();
        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
        assert!(!response.has_more);
    }

    #[test]
    fn items_list_query_default_values() {
        let query = ItemsListQuery::default();
        assert_eq!(query.limit, 20);
        assert_eq!(query.order, SortOrder::Desc);
        assert!(query.after.is_none());
        assert!(query.include.is_none());
    }

    #[test]
    fn items_list_query_to_pagination() {
        let query = ItemsListQuery {
            after: Some("item_123".to_string()),
            limit: 50,
            order: SortOrder::Asc,
            include: Some(vec!["message.input_image.image_url".to_string()]),
        };
        let pagination = query.to_pagination();
        assert_eq!(pagination.after, Some("item_123".to_string()));
        assert_eq!(pagination.limit, 50);
        assert_eq!(pagination.order, SortOrder::Asc);
    }

    #[test]
    fn items_list_query_includes() {
        let query = ItemsListQuery {
            after: None,
            limit: 20,
            order: SortOrder::Desc,
            include: Some(vec![
                "message.input_image.image_url".to_string(),
                "reasoning.encrypted_content".to_string(),
            ]),
        };
        assert!(query.includes("message.input_image.image_url"));
        assert!(query.includes("reasoning.encrypted_content"));
        assert!(!query.includes("message.output_text.logprobs"));
        assert!(query.includes_message_data());
        assert!(query.includes_reasoning_encrypted());
        assert!(!query.includes_computer_call_image());
    }

    #[test]
    fn items_list_query_includes_none() {
        let query = ItemsListQuery::default();
        assert!(!query.includes("message.input_image.image_url"));
        assert!(!query.includes_message_data());
    }
}
