//! Route configuration for Responses API.

use std::sync::Arc;

use axum::{
    Router,
    routing::{delete, get, post},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use super::conversation_handlers::{
    create_conversation, create_items, delete_conversation, delete_item, get_conversation,
    get_item, list_items, update_conversation,
};
use super::handlers::{self, AppState};
use crate::containers;

/// Create the API router with all routes.
pub fn create_router(state: Arc<AppState>) -> Router {
    // CORS configuration - allow all for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Responses API endpoints
        .route("/v1/responses", post(handlers::create_response))
        .route("/v1/responses/{response_id}", get(handlers::get_response))
        .route(
            "/v1/responses/{response_id}",
            delete(handlers::delete_response),
        )
        // Conversations API
        .route("/v1/conversations", post(create_conversation))
        .route(
            "/v1/conversations/{conversation_id}",
            get(get_conversation)
                .post(update_conversation)
                .delete(delete_conversation),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            get(list_items).post(create_items),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(get_item).delete(delete_item),
        )
        // Container file endpoints
        .route(
            "/v1/containers/{container_id}/files/{file_id}/content",
            get(containers::download_file),
        )
        .route(
            "/v1/containers/{container_id}/files",
            get(containers::list_container_files),
        )
        // Compatibility endpoints
        .route("/v1/models", get(handlers::list_models))
        // Health check
        .route("/health", get(handlers::health_check))
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        // Shared state
        .with_state(state)
}
