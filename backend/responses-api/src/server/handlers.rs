//! Request handlers for Responses API endpoints.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Sse, sse::Event},
};
use futures::stream::StreamExt;
use serde_json::json;

use crate::config::Config;
use crate::containers::ContainerStore;
use crate::execution::{
    resolve_chain, ChainResolutionError, ExecutionError, Executor, ExecutorConfig,
    execute_streaming, DEFAULT_MAX_CHAIN_DEPTH,
};
use crate::mcp::McpClient;
use crate::models::{ChatMessage, CreateResponseRequest, DeleteResponse};
use crate::state::{InMemoryStore, ResponseStore};
use crate::translation::assemble_context_from_chain;

/// Shared application state.
pub struct AppState {
    pub executor: Executor,
    pub store: InMemoryStore,
    pub config: Config,
    pub mcp: McpClient,
    pub containers: ContainerStore,
}

/// POST /v1/responses - Create a new response.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Resolve previous_response_id chain if provided
    let (resolved_instructions, previous_messages) =
        resolve_previous_response_chain(&state.store, &req)?;

    // Create a modified request with resolved instructions
    let mut effective_req = req.clone();
    if resolved_instructions.is_some() {
        effective_req.instructions = resolved_instructions;
    }

    if effective_req.stream {
        return create_streaming_response(state, effective_req, previous_messages).await;
    }

    let response = state
        .executor
        .execute(&effective_req, previous_messages)
        .await
        .map_err(execution_error)?;

    if req.store {
        state.store.store(response.clone(), req);
    }

    Ok((StatusCode::OK, Json(response)).into_response())
}

/// Resolve the previous_response_id chain and assemble context.
fn resolve_previous_response_chain(
    store: &InMemoryStore,
    req: &CreateResponseRequest,
) -> Result<(Option<String>, Vec<ChatMessage>), (StatusCode, Json<serde_json::Value>)> {
    if let Some(prev_id) = &req.previous_response_id {
        let chain = resolve_chain(store, prev_id, DEFAULT_MAX_CHAIN_DEPTH)
            .map_err(chain_resolution_error)?;
        Ok(assemble_context_from_chain(&chain, req))
    } else {
        Ok((req.instructions.clone(), vec![]))
    }
}

fn chain_resolution_error(e: ChainResolutionError) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "type": "invalid_request_error",
                "message": e.message
            }
        })),
    )
}

async fn create_streaming_response(
    state: Arc<AppState>,
    req: CreateResponseRequest,
    previous_messages: Vec<ChatMessage>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let executor_config = ExecutorConfig {
        models: state.config.models.clone(),
        max_tool_iterations: state.config.max_tool_iterations,
        timeout_secs: state.config.timeout.as_secs(),
    };

    // Pass store if request wants to store the response
    let store = if req.store {
        Some(state.store.clone())
    } else {
        None
    };

    let mcp = state.mcp.clone();
    let stream = execute_streaming(executor_config, mcp, req, previous_messages, store);

    let sse_stream = stream.map(|result| -> Result<Event, Infallible> {
        match result {
            Ok(sse_event) => {
                let data = serde_json::to_string(&sse_event.data).unwrap_or_default();
                Ok(Event::default().event(sse_event.event).data(data))
            }
            Err(e) => {
                let error_data = json!({
                    "type": "error",
                    "error": {
                        "code": "stream_error",
                        "message": e.to_string()
                    }
                });
                Ok(Event::default().event("error").data(error_data.to_string()))
            }
        }
    });

    Ok(Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

fn execution_error(e: ExecutionError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, error_type) = match &e {
        ExecutionError::Llm(_) => (StatusCode::BAD_GATEWAY, "llm_error"),
        ExecutionError::MaxIterationsExceeded(_) => {
            (StatusCode::UNPROCESSABLE_ENTITY, "max_iterations_exceeded")
        }
        ExecutionError::ModelNotFound(_) => (StatusCode::BAD_REQUEST, "model_not_found"),
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

/// GET /v1/models - List available models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models: Vec<_> = state
        .config
        .models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "created": 1234567890,
                "owned_by": m.owned_by
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({ "object": "list", "data": models })),
    )
}
