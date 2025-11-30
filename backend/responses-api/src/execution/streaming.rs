//! Streaming executor for SSE responses.

use std::pin::Pin;

use eventsource_stream::Eventsource;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::mcp::McpClient;
use crate::models::{
    ChatCompletionChunk, ChatFunctionCall, ChatMessage, ChatRole, ChatToolCall, ChatToolType,
    CreateResponseRequest, FinishReason, FunctionCallOutput, MessageOutput, OutputContent,
    OutputItem, OutputRole, OutputStatus, Response, ResponseStatus, SseEvent, Usage,
};
use crate::translation::{
    function_call_id, message_id, response_id, to_chat_completion, tool_result_message,
};

use super::{ExecutionError, ExecutorConfig};

/// State accumulated during streaming.
#[derive(Default)]
struct StreamState {
    output_index: u32,
    current_message_id: Option<String>,
    accumulated_text: String,
    accumulated_tool_calls: Vec<AccumulatedToolCall>,
    finish_reason: Option<FinishReason>,
}

/// Tool call being accumulated from stream chunks.
#[derive(Clone)]
struct AccumulatedToolCall {
    id: String,
    output_id: String, // The output item ID for this function call
    name: String,
    arguments: String,
}

/// Execute a streaming request, yielding SSE events.
pub fn execute_streaming(
    config: ExecutorConfig,
    mcp: McpClient,
    req: CreateResponseRequest,
) -> Pin<Box<dyn Stream<Item = Result<SseEvent, ExecutionError>> + Send>> {
    let (tx, rx) = mpsc::channel(32);

    tokio::spawn(async move {
        if let Err(e) = run_streaming_loop(config, mcp, req, tx.clone()).await {
            let _ = tx.send(Err(e)).await;
        }
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

async fn run_streaming_loop(
    config: ExecutorConfig,
    mcp: McpClient,
    req: CreateResponseRequest,
    tx: mpsc::Sender<Result<SseEvent, ExecutionError>>,
) -> Result<(), ExecutionError> {
    // Look up model configuration
    let model_config = config
        .models
        .iter()
        .find(|m| m.id == req.model)
        .ok_or_else(|| ExecutionError::ModelNotFound(req.model.clone()))?
        .clone();

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.timeout_secs))
        .build()?;

    let resp_id = response_id();
    let mut conversation: Vec<ChatMessage> = Vec::new();
    let mut iteration = 0;
    let total_input_tokens = 0u32;
    let total_output_tokens = 0u32;

    // Send initial response.created
    let initial_response = build_response(
        &resp_id,
        &req,
        vec![],
        total_input_tokens,
        total_output_tokens,
    );
    send(&tx, SseEvent::response_created(initial_response.clone())).await?;
    send(&tx, SseEvent::response_in_progress(initial_response)).await?;

    loop {
        iteration += 1;
        if iteration > config.max_tool_iterations {
            return Err(ExecutionError::MaxIterationsExceeded(
                config.max_tool_iterations,
            ));
        }

        let mut chat_req = to_chat_completion(&req, Some(conversation.clone()));
        chat_req.stream = true;

        let url = format!("{}/v1/chat/completions", model_config.url);

        // Build request with optional auth
        let mut request = http.post(&url).json(&chat_req);
        if let Some(api_key) = &model_config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::Llm(format!(
                "LLM returned {}: {}",
                status, body
            )));
        }

        let mut state = StreamState::default();

        // Process SSE stream from the backend
        let byte_stream = response.bytes_stream();
        let mut event_stream = byte_stream.eventsource();

        while let Some(event_result) = event_stream.next().await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("SSE parse error: {}", e);
                    continue;
                }
            };

            let data = event.data.trim();
            if data == "[DONE]" {
                break;
            }

            let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to parse chunk: {} - data: {}", e, data);
                    continue;
                }
            };

            for choice in &chunk.choices {
                // Handle text content delta
                if let Some(content) = &choice.delta.content {
                    if state.current_message_id.is_none() {
                        let msg_id = message_id();
                        state.current_message_id = Some(msg_id.clone());

                        // Emit output_item.added for new message
                        let item = OutputItem::Message(MessageOutput {
                            id: msg_id.clone(),
                            status: OutputStatus::InProgress,
                            role: OutputRole::Assistant,
                            content: vec![],
                        });
                        send(
                            &tx,
                            SseEvent::output_item_added(resp_id.clone(), state.output_index, item),
                        )
                        .await?;
                    }

                    state.accumulated_text.push_str(content);

                    // Emit text delta
                    send(
                        &tx,
                        SseEvent::output_text_delta(
                            resp_id.clone(),
                            state.current_message_id.clone().unwrap(),
                            state.output_index,
                            0,
                            content.clone(),
                        ),
                    )
                    .await?;
                }

                // Handle tool call deltas
                if let Some(tool_calls) = &choice.delta.tool_calls {
                    for tc_delta in tool_calls {
                        let tc_index = tc_delta.index as usize;

                        // Ensure we have space for this tool call
                        while state.accumulated_tool_calls.len() <= tc_index {
                            state.accumulated_tool_calls.push(AccumulatedToolCall {
                                id: String::new(),
                                output_id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            });
                        }

                        let tc = &mut state.accumulated_tool_calls[tc_index];

                        if let Some(id) = &tc_delta.id {
                            tc.id = id.clone();
                        }
                        if let Some(func) = &tc_delta.function {
                            if let Some(name) = &func.name {
                                tc.name = name.clone();

                                // Emit function call added (only first time)
                                if tc.output_id.is_empty() {
                                    tc.output_id = function_call_id();
                                }
                                let item = OutputItem::FunctionCall(FunctionCallOutput {
                                    id: tc.output_id.clone(),
                                    call_id: tc.id.clone(),
                                    name: tc.name.clone(),
                                    arguments: String::new(),
                                    status: OutputStatus::InProgress,
                                });
                                send(
                                    &tx,
                                    SseEvent::output_item_added(
                                        resp_id.clone(),
                                        state.output_index + tc_index as u32 + 1,
                                        item,
                                    ),
                                )
                                .await?;
                            }
                            if let Some(args) = &func.arguments {
                                tc.arguments.push_str(args);

                                // Emit arguments delta
                                send(
                                    &tx,
                                    SseEvent::function_call_arguments_delta(
                                        resp_id.clone(),
                                        tc.output_id.clone(),
                                        state.output_index + tc_index as u32 + 1,
                                        args.clone(),
                                    ),
                                )
                                .await?;
                            }
                        }
                    }
                }

                if choice.finish_reason.is_some() {
                    state.finish_reason = choice.finish_reason;
                }
            }
        }

        // Finalize text output if any
        if !state.accumulated_text.is_empty() {
            let msg_id = state.current_message_id.clone().unwrap_or_else(message_id);

            send(
                &tx,
                SseEvent::output_text_done(
                    resp_id.clone(),
                    msg_id.clone(),
                    state.output_index,
                    0,
                    state.accumulated_text.clone(),
                ),
            )
            .await?;

            let item = OutputItem::Message(MessageOutput {
                id: msg_id,
                status: OutputStatus::Completed,
                role: OutputRole::Assistant,
                content: vec![OutputContent::OutputText {
                    text: state.accumulated_text.clone(),
                    annotations: vec![],
                }],
            });
            send(
                &tx,
                SseEvent::output_item_done(resp_id.clone(), state.output_index, item),
            )
            .await?;
        }

        // Handle tool calls if present
        if state.finish_reason == Some(FinishReason::ToolCalls)
            && !state.accumulated_tool_calls.is_empty()
        {
            // Finalize all tool calls
            for (i, tc) in state.accumulated_tool_calls.iter().enumerate() {
                send(
                    &tx,
                    SseEvent::function_call_arguments_done(
                        resp_id.clone(),
                        tc.output_id.clone(),
                        state.output_index + i as u32 + 1,
                        tc.arguments.clone(),
                    ),
                )
                .await?;

                let item = OutputItem::FunctionCall(FunctionCallOutput {
                    id: tc.output_id.clone(),
                    call_id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                    status: OutputStatus::Completed,
                });
                send(
                    &tx,
                    SseEvent::output_item_done(
                        resp_id.clone(),
                        state.output_index + i as u32 + 1,
                        item,
                    ),
                )
                .await?;
            }

            // Build assistant message with tool calls for conversation
            let tool_calls: Vec<ChatToolCall> = state
                .accumulated_tool_calls
                .iter()
                .map(|tc| ChatToolCall {
                    id: tc.id.clone(),
                    tool_type: ChatToolType::Function,
                    function: ChatFunctionCall {
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                })
                .collect();

            conversation.push(ChatMessage {
                role: ChatRole::Assistant,
                content: if state.accumulated_text.is_empty() {
                    None
                } else {
                    Some(crate::models::ChatContent::Text(
                        state.accumulated_text.clone(),
                    ))
                },
                tool_calls: Some(tool_calls),
                tool_call_id: None,
            });

            // Execute tool calls
            for tc in &state.accumulated_tool_calls {
                let arguments: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                let result = match mcp.call_tool_text(&tc.name, arguments).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("Tool {} failed: {}", tc.name, e);
                        format!("Error: {}", e)
                    }
                };

                // Add tool result to conversation
                conversation.push(tool_result_message(tc.id.clone(), result));
            }

            // Continue to next iteration
            continue;
        }

        // No tool calls - we're done
        let final_output = build_final_output(&state);
        let final_response = build_response(
            &resp_id,
            &req,
            final_output,
            total_input_tokens,
            total_output_tokens,
        );

        send(&tx, SseEvent::response_completed(final_response.clone())).await?;
        send(&tx, SseEvent::response_done(final_response)).await?;

        return Ok(());
    }
}

fn build_final_output(state: &StreamState) -> Vec<OutputItem> {
    let mut output = Vec::new();

    if !state.accumulated_text.is_empty() {
        output.push(OutputItem::Message(MessageOutput {
            id: state.current_message_id.clone().unwrap_or_else(message_id),
            status: OutputStatus::Completed,
            role: OutputRole::Assistant,
            content: vec![OutputContent::OutputText {
                text: state.accumulated_text.clone(),
                annotations: vec![],
            }],
        }));
    }

    for tc in &state.accumulated_tool_calls {
        output.push(OutputItem::FunctionCall(FunctionCallOutput {
            id: tc.output_id.clone(),
            call_id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            status: OutputStatus::Completed,
        }));
    }

    output
}

fn build_response(
    id: &str,
    req: &CreateResponseRequest,
    output: Vec<OutputItem>,
    input_tokens: u32,
    output_tokens: u32,
) -> Response {
    Response {
        id: id.to_string(),
        object: Response::OBJECT,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
        status: if output.is_empty() {
            ResponseStatus::InProgress
        } else {
            ResponseStatus::Completed
        },
        error: None,
        incomplete_details: None,
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        model: req.model.clone(),
        output,
        parallel_tool_calls: req.parallel_tool_calls,
        previous_response_id: req.previous_response_id.clone(),
        reasoning: req.reasoning.clone(),
        store: req.store,
        temperature: req.temperature,
        text: req.text.clone(),
        tool_choice: req.tool_choice.clone(),
        tools: req.tools.clone(),
        top_p: req.top_p,
        truncation: req.truncation,
        usage: Usage {
            input_tokens,
            input_tokens_details: None,
            output_tokens,
            output_tokens_details: None,
            total_tokens: input_tokens + output_tokens,
        },
        user: None,
        metadata: req.metadata.clone().unwrap_or(Value::Null),
    }
}

async fn send(
    tx: &mpsc::Sender<Result<SseEvent, ExecutionError>>,
    event: SseEvent,
) -> Result<(), ExecutionError> {
    tx.send(Ok(event))
        .await
        .map_err(|_| ExecutionError::Llm("Stream receiver dropped".to_string()))
}
