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
    Conversation, ConversationDeleted, ConversationItem, ConversationWithItems,
    CreateConversationRequest, CreateItemsQuery, CreateItemsRequest, GetConversationQuery,
    GetItemQuery, ItemsListQuery, ListResponse, PaginationQuery, SortOrder,
    UpdateConversationRequest,
    item_expansion::{ExpansionOptions, expand_item, expand_items},
};

use super::handlers::AppState;

/// Type alias for API error responses.
type ApiError = (StatusCode, Json<serde_json::Value>);

// ============================================================================
// Conversation CRUD
// ============================================================================

/// GET /v1/conversations - List all conversations.
pub async fn list_conversations(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PaginationQuery>,
) -> Result<Json<ListResponse<Conversation>>, ApiError> {
    // Validate query parameters (pagination)
    query
        .validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    let list = state.conversations.list(&query);

    tracing::debug!(
        conversation_count = list.data.len(),
        has_more = list.has_more,
        "Listed conversations"
    );

    Ok(Json(list))
}

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
///
/// Supports `include` query parameter to request additional data:
/// - `include=conversation.items` - Include items in the response
pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(query): Query<GetConversationQuery>,
) -> Result<Json<ConversationWithItems>, ApiError> {
    // Validate include parameter
    query
        .validate_include()
        .map_err(|e| validation_error(&e.to_string()))?;

    let conversation = state
        .conversations
        .get(&conversation_id)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    // Optionally include items if requested
    let items = if query.include_items() {
        let pagination = PaginationQuery {
            after: None,
            limit: 100, // Use a high limit for include
            order: SortOrder::Asc,
        };
        state
            .conversations
            .list_items(&conversation_id, &pagination)
            .map(|list| list.data)
    } else {
        None
    };

    Ok(Json(ConversationWithItems::new(conversation, items)))
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
///
/// Supports `include` query parameter to request additional data in items:
/// - `message.input_image.image_url` - Include full image URLs for input images
/// - `message.output_text.logprobs` - Include log probabilities for output text
/// - `computer_call_output.output.image_url` - Include full image URLs for computer call outputs
/// - `reasoning.encrypted_content` - Include encrypted reasoning content
pub async fn list_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(query): Query<ItemsListQuery>,
) -> Result<Json<ListResponse<ConversationItem>>, ApiError> {
    // Validate query parameters (pagination)
    query
        .validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    // Validate include parameter values
    query
        .validate_include()
        .map_err(|e| validation_error(&e.to_string()))?;

    // Convert to pagination query for storage layer
    let pagination = query.to_pagination();

    let list = state
        .conversations
        .list_items(&conversation_id, &pagination)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    // Expand items based on include parameter
    let mut list = list;
    if query.include.is_some() {
        let options = ExpansionOptions::from(&query);
        let base_url = get_base_url();
        expand_items(&mut list.data, &options, &state.containers, &base_url);

        tracing::debug!(
            conversation_id = %conversation_id,
            include = ?query.include,
            "Items list expanded with include parameter"
        );
    }

    Ok(Json(list))
}

/// POST /v1/conversations/{conversation_id}/items - Add items to a conversation.
///
/// Supports `include` query parameter to request additional data in returned items:
/// - `message.input_image.image_url` - Include full image URLs for input images
/// - `message.output_text.logprobs` - Include log probabilities for output text
/// - `computer_call_output.output.image_url` - Include full image URLs for computer call outputs
/// - `reasoning.encrypted_content` - Include encrypted reasoning content
pub async fn create_items(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
    Query(query): Query<CreateItemsQuery>,
    Json(req): Json<CreateItemsRequest>,
) -> Result<Json<ListResponse<ConversationItem>>, ApiError> {
    // Validate request body
    req.validate()
        .map_err(|e| validation_error(&e.to_string()))?;

    // Validate include parameter values
    query
        .validate_include()
        .map_err(|e| validation_error(&e.to_string()))?;

    let list = state
        .conversations
        .add_items(&conversation_id, req.items)
        .ok_or_else(|| not_found_error("Conversation", &conversation_id))?;

    // Expand items based on include parameter
    let mut list = list;
    if query.include.is_some() {
        let options = ExpansionOptions::from_create_items_query(&query);
        let base_url = get_base_url();
        expand_items(&mut list.data, &options, &state.containers, &base_url);

        tracing::debug!(
            conversation_id = %conversation_id,
            include = ?query.include,
            "Items created with include parameter expansion"
        );
    }

    tracing::info!(
        conversation_id = %conversation_id,
        items_added = list.data.len(),
        "Added items to conversation"
    );

    Ok(Json(list))
}

/// GET /v1/conversations/{conversation_id}/items/{item_id} - Get a single item.
///
/// Supports `include` query parameter to request additional data in the item:
/// - `message.input_image.image_url` - Include full image URLs for input images
/// - `message.output_text.logprobs` - Include log probabilities for output text
/// - `computer_call_output.output.image_url` - Include full image URLs for computer call outputs
/// - `reasoning.encrypted_content` - Include encrypted reasoning content
pub async fn get_item(
    State(state): State<Arc<AppState>>,
    Path((conversation_id, item_id)): Path<(String, String)>,
    Query(query): Query<GetItemQuery>,
) -> Result<Json<ConversationItem>, ApiError> {
    // Validate include parameter values
    query
        .validate_include()
        .map_err(|e| validation_error(&e.to_string()))?;

    // First check if conversation exists
    if state.conversations.get(&conversation_id).is_none() {
        return Err(not_found_error("Conversation", &conversation_id));
    }

    let item = state
        .conversations
        .get_item(&conversation_id, &item_id)
        .ok_or_else(|| not_found_error("Item", &item_id))?;

    // Expand item based on include parameter
    let mut item = item;
    if query.include.is_some() {
        let options = ExpansionOptions::from_get_item_query(&query);
        let base_url = get_base_url();
        expand_item(&mut item, &options, &state.containers, &base_url);

        tracing::debug!(
            conversation_id = %conversation_id,
            item_id = %item_id,
            include = ?query.include,
            "Item retrieved with include parameter expansion"
        );
    }

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

/// Get the base URL for expanding file references.
/// In production, this should come from configuration.
fn get_base_url() -> String {
    std::env::var("RESPONSES_API_BASE_URL").unwrap_or_else(|_| "http://localhost:8000".to_string())
}
