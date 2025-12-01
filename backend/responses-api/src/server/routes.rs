//! Route configuration for Responses API.

use std::sync::Arc;

use axum::{
    Router,
    routing::{delete, get, post},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

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
