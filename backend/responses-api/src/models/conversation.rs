//! Conversation API types.
//!
//! Conversations provide explicit state management for multi-turn interactions,
//! allowing items to be stored and retrieved across Response API calls.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::Metadata;
use super::request::InputItem;
use super::response::OutputStatus;

// ============================================================================
// Conversation Object
// ============================================================================

/// A conversation container for storing conversation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Unique identifier (conv_<uuid>)
    pub id: String,
    /// Object type, always "conversation"
    pub object: &'static str,
    /// Unix timestamp of creation
    pub created_at: i64,
    /// Optional metadata (max 16 key-value pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl Conversation {
    pub const OBJECT: &'static str = "conversation";

    pub fn new(id: String, metadata: Option<Metadata>) -> Self {
        Self {
            id,
            object: Self::OBJECT,
            created_at: chrono_timestamp(),
            metadata,
        }
    }
}

/// A stored item within a conversation.
/// Can be either an input item or an output item from a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    /// Unique item ID (msg_xxx, fc_xxx, rs_xxx, item_xxx)
    pub id: String,
    /// Status of this item
    #[serde(default)]
    pub status: OutputStatus,
    /// The item content (flattened for JSON structure)
    #[serde(flatten)]
    pub content: ConversationItemContent,
}

/// Content of a conversation item - either input or output type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConversationItemContent {
    /// Input item (from request or manually added)
    Input(InputItem),
    /// Output item stored as generic JSON (to preserve all fields)
    Output(Value),
}

// ============================================================================
// Request Types
// ============================================================================

/// Request body for POST /v1/conversations
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CreateConversationRequest {
    /// Initial items to include (max 20)
    #[serde(default)]
    pub items: Option<Vec<InputItem>>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: Option<Metadata>,
}

impl CreateConversationRequest {
    /// Validate the request
    pub fn validate(&self) -> Result<(), ConversationValidationError> {
        if let Some(ref items) = self.items
            && items.len() > 20
        {
            return Err(ConversationValidationError::TooManyItems {
                max: 20,
                actual: items.len(),
            });
        }
        if let Some(ref metadata) = self.metadata {
            metadata.validate().map_err(|e| {
                ConversationValidationError::InvalidMetadata(
                    e.message.map(|m| m.to_string()).unwrap_or_default(),
                )
            })?;
        }
        Ok(())
    }
}

/// Request body for POST /v1/conversations/{id} (update)
#[derive(Debug, Clone, Deserialize)]
pub struct UpdateConversationRequest {
    /// New metadata (replaces existing)
    pub metadata: Metadata,
}

impl UpdateConversationRequest {
    pub fn validate(&self) -> Result<(), ConversationValidationError> {
        self.metadata.validate().map_err(|e| {
            ConversationValidationError::InvalidMetadata(
                e.message.map(|m| m.to_string()).unwrap_or_default(),
            )
        })?;
        Ok(())
    }
}

/// Request body for POST /v1/conversations/{id}/items
#[derive(Debug, Clone, Deserialize)]
pub struct CreateItemsRequest {
    /// Items to add (1-20 items)
    pub items: Vec<InputItem>,
}

impl CreateItemsRequest {
    pub fn validate(&self) -> Result<(), ConversationValidationError> {
        if self.items.is_empty() {
            return Err(ConversationValidationError::NoItems);
        }
        if self.items.len() > 20 {
            return Err(ConversationValidationError::TooManyItems {
                max: 20,
                actual: self.items.len(),
            });
        }
        Ok(())
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for GET /v1/conversations/{id}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GetConversationQuery {
    /// Additional data to include in the response.
    /// Valid values: "conversation.items"
    #[serde(default)]
    pub include: Option<Vec<String>>,
}

impl GetConversationQuery {
    /// Check if items should be included in the response.
    pub fn include_items(&self) -> bool {
        self.include
            .as_ref()
            .map(|v| v.iter().any(|s| s == "conversation.items"))
            .unwrap_or(false)
    }
}

// ============================================================================
// Response Types
// ============================================================================

/// Extended conversation response with optional items.
/// Used when include=["conversation.items"] is specified.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationWithItems {
    /// The conversation object fields
    #[serde(flatten)]
    pub conversation: Conversation,
    /// Items in the conversation (only present when requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<ConversationItem>>,
}

impl ConversationWithItems {
    pub fn new(conversation: Conversation, items: Option<Vec<ConversationItem>>) -> Self {
        Self {
            conversation,
            items,
        }
    }
}

/// Response for DELETE /v1/conversations/{id}
#[derive(Debug, Clone, Serialize)]
pub struct ConversationDeleted {
    pub id: String,
    pub object: &'static str,
    pub deleted: bool,
}

impl ConversationDeleted {
    pub const OBJECT: &'static str = "conversation.deleted";

    pub fn new(id: String) -> Self {
        Self {
            id,
            object: Self::OBJECT,
            deleted: true,
        }
    }
}

// ============================================================================
// Validation Errors
// ============================================================================

/// Validation errors for conversation requests.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConversationValidationError {
    #[error("Too many items: maximum {max}, got {actual}")]
    TooManyItems { max: usize, actual: usize },

    #[error("No items provided")]
    NoItems,

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
}

// ============================================================================
// Helpers
// ============================================================================

/// Get current Unix timestamp in seconds.
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_has_correct_object_type() {
        let conv = Conversation::new("conv_123".to_string(), None);
        assert_eq!(conv.object, "conversation");
    }

    #[test]
    fn conversation_deleted_has_correct_object_type() {
        let deleted = ConversationDeleted::new("conv_123".to_string());
        assert_eq!(deleted.object, "conversation.deleted");
        assert!(deleted.deleted);
    }

    #[test]
    fn create_request_validates_item_limit() {
        let items: Vec<InputItem> = (0..21)
            .map(|_| {
                InputItem::Message(super::super::request::MessageInput {
                    role: super::super::request::Role::User,
                    content: super::super::request::MessageContent::Text("test".to_string()),
                })
            })
            .collect();

        let req = CreateConversationRequest {
            items: Some(items),
            metadata: None,
        };

        assert!(req.validate().is_err());
    }

    #[test]
    fn create_items_request_rejects_empty() {
        let req = CreateItemsRequest { items: vec![] };
        assert!(req.validate().is_err());
    }
}
