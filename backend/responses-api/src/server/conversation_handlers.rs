//! HTTP handlers for the Conversations API.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde_json::json;

use validator::Validate;

use crate::models::{
    Conversation, ConversationDeleted, ConversationItem, CreateConversationRequest,
    CreateItemsRequest, ListResponse, PaginationQuery, UpdateConversationRequest,
};
use crate::state::ConversationStore;

use super::handlers::AppState;

/// Type alias for API error responses.
type ApiError = (StatusCode, Json<serde_json::Value>);

// ============================================================================
// Conversation CRUD
// ============================================================================

/// POST /v1/conversations - Create a new conversation.
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateConversationRequest>,
) -> Result<Json<Conversation>, ApiError> {
    // Validate request
    req.validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    let conversation = state.conversations.create(req.metadata, req.items);

    tracing::info!(
        conversation_id = %conversation.id,
        "Created conversation"
    );

    Ok(Json(conversation))
}

/// GET /v1/conversations/{conversation_id} - Get a conversation.
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Result<Json<Conversation>, ApiError> {
    let conversation = state
        .conversations
        .get(&conversation_id)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    Ok(Json(conversation))
}

/// POST /v1/conversations/{conversation_id} - Update a conversation.
pub async fn update_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(req): Json<UpdateConversationRequest>,
) -> Result<Json<Conversation>, ApiError> {
    // Validate request
    req.validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    let conversation = state
        .conversations
        .update(&conversation_id, req.metadata)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    tracing::info!(
        conversation_id = %conversation_id,
        "Updated conversation metadata"
    );

    Ok(Json(conversation))
}

/// DELETE /v1/conversations/{conversation_id} - Delete a conversation.
pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Result<Json<ConversationDeleted>, ApiError> {
    let deleted = state.conversations.delete(&conversation_id);

    if deleted {
        tracing::info!(
            conversation_id = %conversation_id,
            "Deleted conversation"
        );
        Ok(Json(ConversationDeleted::new(conversation_id)))
    } else {
        Err(not_found_error("Conversation", &conversation_id))
    }
}

// ============================================================================
// Conversation Items
// ============================================================================

/// GET /v1/conversations/{conversation_id}/items - List items in a conversation.
pub async fn list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(query): Query<PaginationQuery>,
) -> Result<Json<ListResponse<ConversationItem>>, ApiError> {
    // Validate pagination parameters
    query
        .validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    let list = state
        .conversations
        .list_items(&conversation_id, &query)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    Ok(Json(list))
}

/// POST /v1/conversations/{conversation_id}/items - Add items to a conversation.
pub async fn create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Json(req): Json<CreateItemsRequest>,
) -> Result<Json<ListResponse<ConversationItem>>, ApiError> {
    // Validate request
    req.validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    let list = state
        .conversations
        .add_items(&conversation_id, req.items)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    tracing::info!(
        conversation_id = %conversation_id,
        items_added = list.data.len(),
        "Added items to conversation"
    );

    Ok(Json(list))
}

/// GET /v1/conversations/{conversation_id}/items/{item_id} - Get a single item.
pub async fn get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Result<Json<ConversationItem>, ApiError> {
    // First check if conversation exists
    if state.conversations.get(&conversation_id).is_none() {
        return Err(not_found_error("Conversation", &conversation_id));
    }

    let item = state
        .conversations
        .get_item(&conversation_id, &item_id)
        .ok_or_else(|| not_found_error("Item", &item_id))?;

    Ok(Json(item))
}

/// DELETE /v1/conversations/{conversation_id}/items/{item_id} - Delete an item.
pub async fn delete_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> Result<Json<Conversation>, ApiError> {
    // First check if conversation exists
    if state.conversations.get(&conversation_id).is_none() {
        return Err(not_found_error("Conversation", &conversation_id));
    }

    let conversation = state
        .conversations
        .delete_item(&conversation_id, &item_id)
        .ok_or_else(|| not_found_error("Item", &item_id))?;

    tracing::info!(
        conversation_id = %conversation_id,
        item_id = %item_id,
        "Deleted item from conversation"
    );

    Ok(Json(conversation))
}

// ============================================================================
// Error Helpers
// ============================================================================

fn validation_error(message: &str) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "type": "invalid_request_error",
                "message": message
            }
        })),
    )
}

fn not_found_error(resource: &str, id: &str) -> ApiError {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "type": "not_found",
                "message": format!("{} {} not found", resource, id)
            }
        })),
    )
}
