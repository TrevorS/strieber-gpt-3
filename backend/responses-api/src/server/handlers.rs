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
    ChainResolutionError, DEFAULT_MAX_CHAIN_DEPTH, ExecutionError, Executor, ExecutorConfig,
    execute_streaming, resolve_chain,
};
use crate::mcp::McpClient;
use crate::models::{ChatMessage, ConversationItem, CreateResponseRequest, DeleteResponse};
use crate::state::{ConversationStore, InMemoryConversationStore, InMemoryStore, ResponseStore};
use crate::translation::assemble_context_from_chain;

/// Type alias for API error responses.
type ApiError = (StatusCode, Json<serde_json::Value>);

/// Shared application state.
pub struct AppState {
    pub executor: Executor,
    pub store: InMemoryStore,
    pub conversations: InMemoryConversationStore,
    pub config: Config,
    pub mcp: McpClient,
    pub containers: ContainerStore,
}

/// POST /v1/responses - Create a new response.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Validate mutual exclusivity of conversation and previous_response_id
    if req.conversation.is_some() && req.previous_response_id.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "Cannot use both 'conversation' and 'previous_response_id' in the same request"
                }
            })),
        ));
    }

    tracing::info!(
        model = %req.model,
        previous_response_id = ?req.previous_response_id,
        conversation_id = ?req.conversation.as_ref().map(|c| &c.id),
        store = req.store,
        stream = req.stream,
        "Creating response"
    );

    // Resolve context from either conversation or previous_response_id chain
    let (resolved_instructions, previous_messages) = if let Some(ref conv_param) = req.conversation
    {
        resolve_conversation_context(&state.conversations, &conv_param.id, &req)?
    } else {
        resolve_previous_response_chain(&state.store, &req)?
    };

    tracing::info!(
        previous_messages_count = previous_messages.len(),
        has_resolved_instructions = resolved_instructions.is_some(),
        "Chain resolved"
    );

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
        state.store.store(response.clone(), req.clone());
    }

    // After creating non-streaming response, append to conversation if specified
    if let Some(ref conv_param) = req.conversation {
        append_response_to_conversation(&state.conversations, &conv_param.id, &req, &response);
    }

    Ok((StatusCode::OK, Json(response)).into_response())
}

/// Resolve the previous_response_id chain and assemble context.
fn resolve_previous_response_chain(
    store: &InMemoryStore,
    req: &CreateResponseRequest,
) -> Result<(Option<String>, Vec<ChatMessage>), ApiError> {
    if let Some(prev_id) = &req.previous_response_id {
        tracing::debug!(
            previous_response_id = %prev_id,
            store_count = store.len(),
            "Resolving chain from previous_response_id"
        );

        let chain = resolve_chain(store, prev_id, DEFAULT_MAX_CHAIN_DEPTH)
            .map_err(chain_resolution_error)?;

        tracing::info!(
            chain_length = chain.len(),
            chain_ids = ?chain.iter().map(|s| &s.response.id).collect::<Vec<_>>(),
            "Chain resolved successfully"
        );

        let (instructions, messages) = assemble_context_from_chain(&chain, req);

        // Log message contents for debugging
        for (i, msg) in messages.iter().enumerate() {
            let content_preview = match &msg.content {
                Some(crate::models::ChatContent::Text(t)) => {
                    if t.len() > 100 {
                        format!("{}...", &t[..100])
                    } else {
                        t.clone()
                    }
                }
                Some(crate::models::ChatContent::Parts(_)) => "[parts]".to_string(),
                None => "[none]".to_string(),
            };
            tracing::debug!(
                index = i,
                role = ?msg.role,
                content_preview = %content_preview,
                "Previous message"
            );
        }

        Ok((instructions, messages))
    } else {
        tracing::debug!("No previous_response_id provided, starting fresh conversation");
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

    // Extract conversation parameters for streaming
    let conversation_id = req.conversation.as_ref().map(|c| c.id.clone());
    let conversation_store = if conversation_id.is_some() {
        Some(state.conversations.clone())
    } else {
        None
    };

    let mcp = state.mcp.clone();
    let containers = state.containers.clone();
    let stream = execute_streaming(
        executor_config,
        mcp,
        req,
        previous_messages,
        store,
        containers,
        conversation_store,
        conversation_id,
    );

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
                "owned_by": m.owned_by,
                "supports_vision": m.supports_vision,
                "supported_tools": m.supported_tools
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({ "object": "list", "data": models })),
    )
}

// ============================================================================
// Conversation Integration Helpers
// ============================================================================

/// Resolve conversation context into chat messages.
fn resolve_conversation_context(
    store: &InMemoryConversationStore,
    conversation_id: &str,
    req: &CreateResponseRequest,
) -> Result<(Option<String>, Vec<ChatMessage>), ApiError> {
    use crate::models::{PaginationQuery, SortOrder};

    // Check if conversation exists
    let _conversation = store.get(conversation_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Conversation {} not found", conversation_id)
                }
            })),
        )
    })?;

    // Get all items from conversation (use high limit, asc order)
    let query = PaginationQuery {
        after: None,
        limit: 100,
        order: SortOrder::Asc,
    };

    let items_response = store.list_items(conversation_id, &query);
    let items = items_response.map(|r| r.data).unwrap_or_default();

    // Convert conversation items to chat messages
    let messages = conversation_items_to_chat_messages(&items);

    tracing::info!(
        conversation_id = %conversation_id,
        items_count = items.len(),
        messages_count = messages.len(),
        "Loaded conversation context"
    );

    Ok((req.instructions.clone(), messages))
}

/// Convert conversation items to chat messages for context.
fn conversation_items_to_chat_messages(items: &[ConversationItem]) -> Vec<ChatMessage> {
    use crate::models::{
        ChatContent, ChatRole, ConversationItemContent, InputItem, MessageContent,
    };

    items
        .iter()
        .filter_map(|item| {
            match &item.content {
                ConversationItemContent::Input(input_item) => {
                    match input_item {
                        InputItem::Message(msg) => {
                            let role = match msg.role {
                                crate::models::Role::User => ChatRole::User,
                                crate::models::Role::Assistant => ChatRole::Assistant,
                                crate::models::Role::System | crate::models::Role::Developer => {
                                    ChatRole::System
                                }
                            };
                            let content = match &msg.content {
                                MessageContent::Text(t) => Some(ChatContent::Text(t.clone())),
                                MessageContent::Parts(_) => None, // TODO: handle parts
                            };
                            content.map(|c| ChatMessage {
                                role,
                                content: Some(c),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: None,
                            })
                        }
                        _ => None, // Skip other input types for now
                    }
                }
                ConversationItemContent::Output(output_json) => {
                    // Parse output JSON and extract message content
                    // Output items have a "type" field to identify them
                    let type_field = output_json.get("type").and_then(|t| t.as_str());

                    match type_field {
                        Some("message") => {
                            // Extract text content from message output
                            let content_array =
                                output_json.get("content").and_then(|c| c.as_array());
                            let text = content_array.and_then(|arr| {
                                arr.iter().find_map(|part| {
                                    if part.get("type").and_then(|t| t.as_str())
                                        == Some("output_text")
                                    {
                                        part.get("text")
                                            .and_then(|t| t.as_str())
                                            .map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                })
                            });

                            text.map(|t| ChatMessage {
                                role: ChatRole::Assistant,
                                content: Some(ChatContent::Text(t)),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: None,
                            })
                        }
                        Some("reasoning") => {
                            // Extract reasoning content and wrap in think tags
                            let content_array =
                                output_json.get("content").and_then(|c| c.as_array());
                            let text = content_array.and_then(|arr| {
                                let texts: Vec<String> = arr
                                    .iter()
                                    .filter_map(|part| {
                                        if part.get("type").and_then(|t| t.as_str())
                                            == Some("reasoning_text")
                                        {
                                            part.get("text")
                                                .and_then(|t| t.as_str())
                                                .map(|s| s.to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                if texts.is_empty() {
                                    None
                                } else {
                                    Some(texts.join(""))
                                }
                            });

                            text.map(|t| ChatMessage {
                                role: ChatRole::Assistant,
                                content: None,
                                reasoning_content: Some(t),
                                tool_calls: None,
                                tool_call_id: None,
                            })
                        }
                        _ => None, // Skip function calls and other types
                    }
                }
            }
        })
        .collect()
}

/// Append request input and response output to conversation.
fn append_response_to_conversation(
    store: &InMemoryConversationStore,
    conversation_id: &str,
    req: &CreateResponseRequest,
    response: &crate::models::Response,
) {
    use crate::models::{ConversationItemContent, Input, InputItem, OutputStatus};
    use crate::translation::{function_call_id, item_id, message_id, reasoning_id};

    let mut all_items: Vec<ConversationItem> = Vec::new();

    // First, append input items from the request
    let input_items: Vec<InputItem> = match &req.input {
        Input::Empty => vec![],
        Input::Text(text) => {
            // Convert simple text to a user message
            vec![InputItem::Message(crate::models::MessageInput {
                role: crate::models::Role::User,
                content: crate::models::MessageContent::Text(text.clone()),
            })]
        }
        Input::Items(items) => items.clone(),
    };

    for input_item in input_items {
        let id = match &input_item {
            InputItem::Message(_) => message_id(),
            InputItem::Reasoning(_) => reasoning_id(),
            InputItem::FunctionCall(_) => function_call_id(),
            _ => item_id(),
        };
        all_items.push(ConversationItem {
            id,
            status: OutputStatus::Completed,
            content: ConversationItemContent::Input(input_item),
        });
    }

    // Then, append output items from the response
    for output in &response.output {
        let id = match output {
            crate::models::OutputItem::Message(_) => message_id(),
            crate::models::OutputItem::FunctionCall(_) => function_call_id(),
            crate::models::OutputItem::Reasoning(_) => reasoning_id(),
            _ => item_id(),
        };
        all_items.push(ConversationItem {
            id,
            status: OutputStatus::Completed,
            content: ConversationItemContent::Output(
                serde_json::to_value(output).unwrap_or_default(),
            ),
        });
    }

    if !all_items.is_empty() {
        let input_count = all_items
            .iter()
            .filter(|i| matches!(i.content, ConversationItemContent::Input(_)))
            .count();
        let output_count = all_items.len() - input_count;

        store.append_output_items(conversation_id, all_items);
        tracing::debug!(
            conversation_id = %conversation_id,
            input_items_appended = input_count,
            output_items_appended = output_count,
            "Appended request and response to conversation"
        );
    }
}
