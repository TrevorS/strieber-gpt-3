//! Request handlers for Responses API endpoints.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde_json::json;

use crate::execution::{ExecutionError, Executor};
use crate::models::{CreateResponseRequest, DeleteResponse};
use crate::state::{InMemoryStore, ResponseStore};

/// Shared application state.
pub struct AppState {
    pub executor: Executor,
    pub store: InMemoryStore,
}

/// POST /v1/responses - Create a new response.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Check if streaming is requested
    if req.stream {
        // TODO: Implement SSE streaming
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "type": "not_implemented",
                    "message": "Streaming not yet supported"
                }
            })),
        ));
    }

    // Execute the request
    let response = state.executor.execute(&req).await.map_err(|e| {
        let (status, error_type) = match &e {
            ExecutionError::Llm(_) => (StatusCode::BAD_GATEWAY, "llm_error"),
            ExecutionError::MaxIterationsExceeded(_) => {
                (StatusCode::UNPROCESSABLE_ENTITY, "max_iterations_exceeded")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };
        (
            status,
            Json(json!({
                "error": {
                    "type": error_type,
                    "message": e.to_string()
                }
            })),
        )
    })?;

    // Store the response if store=true
    if req.store {
        state.store.store(response.clone(), req);
    }

    Ok((StatusCode::OK, Json(response)))
}

/// GET /v1/responses/{response_id} - Get a response by ID.
pub async fn get_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let stored = state.store.get(&response_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Response {} not found", response_id)
                }
            })),
        )
    })?;

    Ok((StatusCode::OK, Json(stored.response)))
}

/// DELETE /v1/responses/{response_id} - Delete a response.
pub async fn delete_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let deleted = state.store.delete(&response_id);

    if deleted {
        Ok((
            StatusCode::OK,
            Json(DeleteResponse {
                id: response_id,
                object: DeleteResponse::OBJECT,
                deleted: true,
            }),
        ))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Response {} not found", response_id)
                }
            })),
        ))
    }
}

/// GET /health - Health check endpoint.
pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

/// GET /v1/models - List available models (for compatibility).
pub async fn list_models() -> impl IntoResponse {
    // Return the model that llama.cpp is serving
    // In practice, this should be configurable or fetched from llama.cpp
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [
                {
                    "id": "gpt-oss-120b",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "local"
                }
            ]
        })),
    )
}
