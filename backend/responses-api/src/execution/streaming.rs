//! Streaming executor for SSE responses.

use std::pin::Pin;

use eventsource_stream::Eventsource;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;

use rmcp::model::RawContent;

use crate::containers::ContainerStore;
use crate::mcp::{CallToolResult, McpClient};
use crate::models::{
    Annotation, ChatCompletionChunk, ChatFunctionCall, ChatMessage, ChatRole, ChatToolCall,
    ChatToolType, ConversationItem, ConversationItemContent, CreateResponseRequest, FinishReason,
    FunctionCallOutput, FunctionToolWrapper, MessageOutput, OutputContent, OutputItem, OutputRole,
    OutputStatus, ReasoningContent, ReasoningOutput, Response, ResponseStatus, SseEvent, Tool,
    Usage, WebSearchAction, WebSearchCallOutput, WebSearchSource,
};
use crate::state::{ConversationStore, InMemoryConversationStore, InMemoryStore, ResponseStore};
use crate::translation::{
    build_url_citations, function_call_id, message_id, parse_reasoning_tags, reasoning_id,
    response_id, to_chat_completion, tool_result_message,
};

use super::{ExecutionError, ExecutorConfig, GeneratedFile};

/// State accumulated during streaming.
#[derive(Default)]
struct StreamState {
    output_index: u32,
    current_message_id: Option<String>,
    current_reasoning_id: Option<String>,
    accumulated_text: String,
    accumulated_reasoning: String,
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
///
/// # Arguments
///
/// * `config` - Executor configuration
/// * `mcp` - MCP client for tool calls
/// * `req` - The request to execute
/// * `previous_messages` - Messages from resolved previous_response_id chain
/// * `store` - Optional store for persisting the response (if req.store is true)
/// * `containers` - Container store for persisting generated files (images, etc.)
/// * `conversation_store` - Optional conversation store for stateful conversations
/// * `conversation_id` - Optional conversation ID to append output items to
#[allow(clippy::too_many_arguments)]
pub fn execute_streaming(
    config: ExecutorConfig,
    mcp: McpClient,
    req: CreateResponseRequest,
    previous_messages: Vec<ChatMessage>,
    store: Option<InMemoryStore>,
    containers: ContainerStore,
    conversation_store: Option<InMemoryConversationStore>,
    conversation_id: Option<String>,
) -> Pin<Box<dyn Stream<Item = Result<SseEvent, ExecutionError>> + Send>> {
    let (tx, rx) = mpsc::channel(32);

    tokio::spawn(async move {
        if let Err(e) = run_streaming_loop(
            config,
            mcp,
            req,
            previous_messages,
            store,
            containers,
            conversation_store,
            conversation_id,
            tx.clone(),
        )
        .await
        {
            let _ = tx.send(Err(e)).await;
        }
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming_loop(
    config: ExecutorConfig,
    mcp: McpClient,
    req: CreateResponseRequest,
    previous_messages: Vec<ChatMessage>,
    store: Option<InMemoryStore>,
    containers: ContainerStore,
    conversation_store: Option<InMemoryConversationStore>,
    conversation_id: Option<String>,
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

    // Expand built-in tools to function definitions
    let expanded_tools = expand_tools(&mcp, &req.tools).await;
    let mut req = req;
    req.tools = expanded_tools;

    // Apply model's default reasoning config if request doesn't specify one
    if req.reasoning.is_none()
        && let Some(ref default_reasoning) = model_config.reasoning
    {
        tracing::info!(
            model = %req.model,
            effort = ?default_reasoning.effort,
            "Applying default reasoning config from model"
        );
        req.reasoning = Some(default_reasoning.clone());
    }

    // Extract attached images from input for tool injection
    use crate::translation::{
        AttachedImage, extract_attached_images, replace_images_with_placeholders,
    };
    let attached_images: Vec<AttachedImage> = extract_attached_images(&req.input);
    if !attached_images.is_empty() {
        tracing::info!(
            image_count = attached_images.len(),
            model_supports_vision = model_config.supports_vision,
            "Found attached images in request"
        );
    }

    // For non-vision models, replace images with text placeholders
    // so the LLM knows images are attached but doesn't receive the raw data
    let effective_input = if !attached_images.is_empty() && !model_config.supports_vision {
        tracing::info!("Replacing images with placeholders for non-vision model");
        replace_images_with_placeholders(&req.input)
    } else {
        req.input.clone()
    };

    // Initialize conversation with previous messages from chain
    let mut conversation: Vec<ChatMessage> = previous_messages;

    // Add the current request's user input to the conversation
    // This ensures it appears BEFORE any tool calls/results in subsequent iterations
    use crate::translation::input_to_messages;
    conversation.extend(input_to_messages(&effective_input));

    let mut iteration = 0;
    let total_input_tokens = 0u32;
    let total_output_tokens = 0u32;

    tracing::info!(
        response_id = %resp_id,
        model = %req.model,
        previous_messages_count = conversation.len(),
        "Starting streaming execution"
    );

    // Log each previous message for debugging
    for (i, msg) in conversation.iter().enumerate() {
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
            has_tool_calls = msg.tool_calls.is_some(),
            "Previous message in conversation"
        );
    }

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

    // Web search sources must persist across loop iterations (tool call cycles)
    // because sources are extracted during tool execution but used for annotations
    // in the final message after all tool calls complete.
    let mut persistent_web_search_sources: Vec<WebSearchSource> = Vec::new();

    // Generated files (images, etc.) must persist across tool call cycles for annotations
    let mut persistent_generated_files: Vec<GeneratedFile> = Vec::new();
    // Create a container for storing generated files (images, etc.)
    let container_id = containers.create();
    tracing::debug!(container_id = %container_id, "Created container for response");

    loop {
        iteration += 1;
        if iteration > config.max_tool_iterations {
            tracing::warn!(
                max_iterations = config.max_tool_iterations,
                generated_files = persistent_generated_files.len(),
                "Max tool iterations exceeded, sending response with accumulated data"
            );

            // Build final output with accumulated files (even without accumulated text)
            // This ensures generated images are delivered even when model loops
            let final_output = build_final_output_graceful(
                &persistent_web_search_sources,
                &persistent_generated_files,
            );
            let final_response = build_response(
                &resp_id,
                &req,
                final_output,
                total_input_tokens,
                total_output_tokens,
            );

            // Store the response if requested
            if req.store
                && let Some(ref store) = store
            {
                store.store(final_response.clone(), req.clone());
                tracing::info!(
                    response_id = %resp_id,
                    output_items = final_response.output.len(),
                    "Stored streaming response (max iterations)"
                );
            }

            // Append input and output to conversation if using conversation API
            if let (Some(conv_store), Some(conv_id)) = (&conversation_store, &conversation_id) {
                append_to_conversation(conv_store, conv_id, &req, &final_response.output);
            }

            send(&tx, SseEvent::response_completed(final_response.clone())).await?;
            send(&tx, SseEvent::response_done(final_response)).await?;

            return Ok(());
        }

        let mut chat_req = to_chat_completion(&req, Some(conversation.clone()));
        chat_req.stream = true;

        tracing::info!(
            iteration,
            total_messages = chat_req.messages.len(),
            has_tools = chat_req.tools.is_some(),
            "Sending request to LLM"
        );

        // Log all messages being sent to LLM
        for (i, msg) in chat_req.messages.iter().enumerate() {
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
                content = %content_preview,
                "Message to LLM"
            );
        }

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
                            state.current_message_id.clone().unwrap_or_else(message_id),
                            state.output_index,
                            0,
                            content.clone(),
                        ),
                    )
                    .await?;
                }

                // Handle reasoning_content delta (gpt-oss, DeepSeek-R1, etc.)
                if let Some(reasoning) = &choice.delta.reasoning_content {
                    if state.current_reasoning_id.is_none() {
                        let reason_id = reasoning_id();
                        state.current_reasoning_id = Some(reason_id.clone());

                        // Emit output_item.added for new reasoning
                        let item = OutputItem::Reasoning(ReasoningOutput {
                            id: reason_id.clone(),
                            status: OutputStatus::InProgress,
                            content: vec![],
                            summary: vec![],
                            encrypted_content: None,
                        });
                        send(
                            &tx,
                            SseEvent::output_item_added(resp_id.clone(), state.output_index, item),
                        )
                        .await?;
                        state.output_index += 1;
                    }

                    state.accumulated_reasoning.push_str(reasoning);

                    // Emit reasoning delta
                    send(
                        &tx,
                        SseEvent::reasoning_delta(
                            resp_id.clone(),
                            state
                                .current_reasoning_id
                                .clone()
                                .unwrap_or_else(reasoning_id),
                            state.output_index - 1, // Use the reasoning item's index
                            0,
                            reasoning.clone(),
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
                                let item = if tc.name == "web_search" {
                                    OutputItem::WebSearchCall(WebSearchCallOutput {
                                        id: tc.output_id.clone(),
                                        status: OutputStatus::InProgress,
                                        action: None, // No sources yet during streaming
                                    })
                                } else {
                                    OutputItem::FunctionCall(FunctionCallOutput {
                                        id: tc.output_id.clone(),
                                        call_id: tc.id.clone(),
                                        name: tc.name.clone(),
                                        arguments: String::new(),
                                        status: OutputStatus::InProgress,
                                        output: None,
                                    })
                                };
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

        // Finalize reasoning from reasoning_content field (gpt-oss, DeepSeek-R1, etc.)
        // This was streamed via reasoning_delta events, now we complete it
        if !state.accumulated_reasoning.is_empty()
            && let Some(reason_id) = &state.current_reasoning_id
        {
            let reasoning_item = OutputItem::Reasoning(ReasoningOutput {
                id: reason_id.clone(),
                status: OutputStatus::Completed,
                content: vec![ReasoningContent::ReasoningText {
                    text: state.accumulated_reasoning.clone(),
                }],
                summary: vec![],
                encrypted_content: None,
            });
            // Emit done for the reasoning item we already added during streaming
            // The output_index for reasoning was incremented when we added it
            send(
                &tx,
                SseEvent::output_item_done(
                    resp_id.clone(),
                    0, // Reasoning is always first output item
                    reasoning_item,
                ),
            )
            .await?;
        }

        // Finalize text output if any
        if !state.accumulated_text.is_empty() {
            // Parse reasoning tags from accumulated text (fallback for models using <think> tags)
            // Skip if we already got reasoning from reasoning_content field
            let (reasoning_text, clean_text) = if state.accumulated_reasoning.is_empty() {
                parse_reasoning_tags(&state.accumulated_text)
            } else {
                (None, state.accumulated_text.clone())
            };
            let mut current_output_index = state.output_index;

            // Emit reasoning output item if present (only for <think> tag models)
            if let Some(reasoning) = reasoning_text {
                let reasoning_item = OutputItem::Reasoning(ReasoningOutput {
                    id: reasoning_id(),
                    status: OutputStatus::Completed,
                    content: vec![ReasoningContent::ReasoningText { text: reasoning }],
                    summary: vec![],
                    encrypted_content: None,
                });
                // Emit added and done for reasoning
                send(
                    &tx,
                    SseEvent::output_item_added(
                        resp_id.clone(),
                        current_output_index,
                        reasoning_item.clone(),
                    ),
                )
                .await?;
                send(
                    &tx,
                    SseEvent::output_item_done(
                        resp_id.clone(),
                        current_output_index,
                        reasoning_item,
                    ),
                )
                .await?;
                current_output_index += 1;
            }

            // Emit message with clean text (without <think> tags)
            if !clean_text.is_empty() {
                let msg_id = state.current_message_id.clone().unwrap_or_else(message_id);

                send(
                    &tx,
                    SseEvent::output_text_done(
                        resp_id.clone(),
                        msg_id.clone(),
                        current_output_index,
                        0,
                        clean_text.clone(),
                    ),
                )
                .await?;

                // Build URL citation annotations from accumulated web search sources
                let annotations = if !persistent_web_search_sources.is_empty() {
                    build_url_citations(&clean_text, &persistent_web_search_sources)
                } else {
                    vec![]
                };

                if !annotations.is_empty() {
                    tracing::debug!(
                        annotation_count = annotations.len(),
                        "Including URL citation annotations in streaming output"
                    );
                }

                let item = OutputItem::Message(MessageOutput {
                    id: msg_id,
                    status: OutputStatus::Completed,
                    role: OutputRole::Assistant,
                    content: vec![OutputContent::OutputText {
                        text: clean_text,
                        annotations,
                    }],
                });
                send(
                    &tx,
                    SseEvent::output_item_done(resp_id.clone(), current_output_index, item),
                )
                .await?;
            }
        }

        // Handle tool calls if present
        if state.finish_reason == Some(FinishReason::ToolCalls)
            && !state.accumulated_tool_calls.is_empty()
        {
            // Finalize all tool calls (except web_search which needs sources from execution)
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

                // Note: output_item_done is sent AFTER tool execution completes
                // so the UI shows the checkmark only when the tool finishes
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
                reasoning_content: None,
                tool_calls: Some(tool_calls),
                tool_call_id: None,
            });

            // Execute tool calls
            for (i, tc) in state.accumulated_tool_calls.iter().enumerate() {
                let mut arguments: Value =
                    serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);

                // Inject attached images into tool arguments
                // If the tool has an "image_data" parameter with value like "image_0" or "attached",
                // replace it with the actual base64 image data
                if let Some(obj) = arguments.as_object_mut()
                    && let Some(image_param) = obj.get("image_data").cloned()
                    && let Some(image_ref) = image_param.as_str()
                {
                    // Handle "attached" (use first image) or "image_N" format
                    let image_index = if image_ref == "attached" {
                        Some(0)
                    } else if image_ref.starts_with("image_") {
                        image_ref
                            .strip_prefix("image_")
                            .and_then(|n| n.parse::<usize>().ok())
                    } else {
                        None
                    };

                    if let Some(idx) = image_index {
                        if let Some(img) = attached_images.get(idx) {
                            tracing::info!(
                                tool = %tc.name,
                                image_id = %img.id,
                                data_len = img.data.len(),
                                "Injecting attached image into tool call"
                            );
                            obj.insert("image_data".to_string(), Value::String(img.data.clone()));
                        } else {
                            tracing::warn!(
                                tool = %tc.name,
                                requested_index = idx,
                                available_images = attached_images.len(),
                                "Requested image index not found"
                            );
                        }
                    }
                }

                let result_text = match mcp.call_tool(&tc.name, arguments.clone()).await {
                    Ok(tool_result) => {
                        // Extract text and images from result
                        let (text, generated_files) = extract_content_from_result(
                            &tool_result,
                            &container_id,
                            &containers,
                            persistent_generated_files.len(),
                        );

                        // Track generated files for annotations
                        if !generated_files.is_empty() {
                            tracing::info!(
                                tool = %tc.name,
                                file_count = generated_files.len(),
                                "Generated files from tool execution"
                            );
                            persistent_generated_files.extend(generated_files);
                        }

                        // If web_search tool, extract sources for citation annotations
                        if tc.name.contains("web_search") || tc.name.contains("news_search") {
                            // Log what we got from MCP
                            tracing::info!(
                                tool = %tc.name,
                                has_structured_content = tool_result.structured_content.is_some(),
                                has_meta = tool_result.meta.is_some(),
                                "Checking MCP result for sources"
                            );

                            let sources = extract_sources_from_result(&tool_result);

                            // Send output_item_done for web_search with sources
                            let item = OutputItem::WebSearchCall(WebSearchCallOutput {
                                id: tc.output_id.clone(),
                                status: OutputStatus::Completed,
                                action: Some(WebSearchAction {
                                    query: extract_query_from_args(&tc.arguments),
                                    sources: sources.clone(),
                                }),
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

                            if !sources.is_empty() {
                                tracing::info!(
                                    tool = %tc.name,
                                    source_count = sources.len(),
                                    "Extracted web search sources for citations"
                                );
                                persistent_web_search_sources.extend(sources);
                            } else {
                                tracing::info!(
                                    tool = %tc.name,
                                    "No sources extracted from tool result"
                                );
                            }
                        }

                        text
                    }
                    Err(e) => {
                        tracing::error!("Tool {} failed: {}", tc.name, e);
                        format!("Error: {}", e)
                    }
                };

                // Send output_item_done for non-web_search tools AFTER execution completes
                // (web_search tools send their own output_item_done above with sources)
                if !tc.name.contains("web_search") && !tc.name.contains("news_search") {
                    let item = OutputItem::FunctionCall(FunctionCallOutput {
                        id: tc.output_id.clone(),
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                        status: OutputStatus::Completed,
                        output: Some(result_text.clone()),
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

                // Add tool result to conversation
                conversation.push(tool_result_message(tc.id.clone(), result_text));
            }

            // Continue to next iteration
            continue;
        }

        // No tool calls - we're done
        let final_output = build_final_output(
            &state,
            &persistent_web_search_sources,
            &persistent_generated_files,
        );
        let final_response = build_response(
            &resp_id,
            &req,
            final_output,
            total_input_tokens,
            total_output_tokens,
        );

        // Store the response if requested
        if req.store {
            if let Some(ref store) = store {
                store.store(final_response.clone(), req.clone());
                tracing::info!(
                    response_id = %resp_id,
                    previous_response_id = ?req.previous_response_id,
                    output_items = final_response.output.len(),
                    store_size = store.len(),
                    "Stored streaming response"
                );
            }
        } else {
            tracing::debug!(response_id = %resp_id, "Response not stored (store=false)");
        }

        // Append input and output to conversation if using conversation API
        if let (Some(conv_store), Some(conv_id)) = (&conversation_store, &conversation_id) {
            append_to_conversation(conv_store, conv_id, &req, &final_response.output);

            // Generate title for new conversations (first exchange only)
            if super::title_generator::should_generate_title(conv_store, conv_id) {
                if let Some(task_model) = super::title_generator::find_task_model(&config.models) {
                    let user_msg = super::title_generator::extract_first_user_message(&req.input);
                    let assistant_resp =
                        super::title_generator::extract_assistant_response(&final_response.output);

                    match super::title_generator::generate_title(
                        task_model,
                        &http,
                        &user_msg,
                        &assistant_resp,
                    )
                    .await
                    {
                        Ok(title) => {
                            // Update conversation title in store
                            if let Err(e) = conv_store.update_title(conv_id, &title) {
                                tracing::warn!(
                                    conversation_id = %conv_id,
                                    error = %e,
                                    "Failed to update conversation title in store"
                                );
                            }
                            // Emit title event to frontend
                            send(
                                &tx,
                                SseEvent::conversation_title_generated(conv_id.clone(), title),
                            )
                            .await?;
                        }
                        Err(e) => {
                            tracing::warn!(
                                conversation_id = %conv_id,
                                error = %e,
                                "Failed to generate conversation title"
                            );
                        }
                    }
                } else {
                    tracing::debug!("No task model configured, skipping title generation");
                }
            }
        }

        send(&tx, SseEvent::response_completed(final_response.clone())).await?;
        send(&tx, SseEvent::response_done(final_response)).await?;

        return Ok(());
    }
}

fn build_final_output(
    state: &StreamState,
    web_search_sources: &[WebSearchSource],
    generated_files: &[GeneratedFile],
) -> Vec<OutputItem> {
    let mut output = Vec::new();

    if !state.accumulated_text.is_empty() {
        // Parse reasoning tags from accumulated text
        let (reasoning_text, clean_text) = parse_reasoning_tags(&state.accumulated_text);

        // Emit reasoning first if present
        if let Some(reasoning) = reasoning_text {
            output.push(OutputItem::Reasoning(ReasoningOutput {
                id: reasoning_id(),
                status: OutputStatus::Completed,
                content: vec![ReasoningContent::ReasoningText { text: reasoning }],
                summary: vec![],
                encrypted_content: None,
            }));
        }

        // Then emit message with clean text (without <think> tags)
        if !clean_text.is_empty() {
            // Build annotations from web search sources and generated files
            let mut annotations = if !web_search_sources.is_empty() {
                build_url_citations(&clean_text, web_search_sources)
            } else {
                vec![]
            };

            // Add file citation annotations for generated images
            for file in generated_files {
                annotations.push(Annotation::ContainerFileCitation {
                    container_id: file.container_id.clone(),
                    file_id: file.file_id.clone(),
                    filename: file.filename.clone(),
                });
            }

            if !annotations.is_empty() {
                tracing::debug!(
                    annotation_count = annotations.len(),
                    source_count = web_search_sources.len(),
                    file_count = generated_files.len(),
                    "Built annotations for message"
                );
            }

            output.push(OutputItem::Message(MessageOutput {
                id: state.current_message_id.clone().unwrap_or_else(message_id),
                status: OutputStatus::Completed,
                role: OutputRole::Assistant,
                content: vec![OutputContent::OutputText {
                    text: clean_text,
                    annotations,
                }],
            }));
        }
    }

    // Fallback: If we have generated files but no message was created, create one with just annotations
    // This handles the case where the model only outputs reasoning (no user-visible text) but generates images
    if !generated_files.is_empty() && !output.iter().any(|o| matches!(o, OutputItem::Message(_))) {
        let annotations: Vec<Annotation> = generated_files
            .iter()
            .map(|file| Annotation::ContainerFileCitation {
                container_id: file.container_id.clone(),
                file_id: file.file_id.clone(),
                filename: file.filename.clone(),
            })
            .collect();

        tracing::info!(
            file_count = generated_files.len(),
            "Creating message with file citations (no text output)"
        );

        output.push(OutputItem::Message(MessageOutput {
            id: message_id(),
            status: OutputStatus::Completed,
            role: OutputRole::Assistant,
            content: vec![OutputContent::OutputText {
                text: String::new(), // Empty text, annotations carry the images
                annotations,
            }],
        }));
    }

    for tc in &state.accumulated_tool_calls {
        if tc.name.contains("web_search") || tc.name.contains("news_search") {
            // Emit WebSearchCall for web_search tools
            output.push(OutputItem::WebSearchCall(WebSearchCallOutput {
                id: tc.output_id.clone(),
                status: OutputStatus::Completed,
                action: Some(WebSearchAction {
                    query: extract_query_from_args(&tc.arguments),
                    sources: web_search_sources.to_vec(),
                }),
            }));
        } else {
            output.push(OutputItem::FunctionCall(FunctionCallOutput {
                id: tc.output_id.clone(),
                call_id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
                status: OutputStatus::Completed,
                output: None, // Output was sent in streaming event
            }));
        }
    }

    output
}

/// Build final output when max iterations is exceeded.
/// Creates a message with file citations even if there's no accumulated text.
fn build_final_output_graceful(
    web_search_sources: &[WebSearchSource],
    generated_files: &[GeneratedFile],
) -> Vec<OutputItem> {
    let mut output = Vec::new();

    // If we have generated files, create a message with just the file citations
    // This ensures images are delivered to the frontend even when model loops
    if !generated_files.is_empty() {
        let mut annotations: Vec<Annotation> = Vec::new();

        // Add file citation annotations for generated images
        for file in generated_files {
            annotations.push(Annotation::ContainerFileCitation {
                container_id: file.container_id.clone(),
                file_id: file.file_id.clone(),
                filename: file.filename.clone(),
            });
        }

        // Add web search citations if any
        if !web_search_sources.is_empty() {
            // For graceful fallback, we don't have text to match citations to
            // Just add sources as-is (they'll appear as references)
            for source in web_search_sources {
                annotations.push(Annotation::UrlCitation {
                    start_index: 0,
                    end_index: 0,
                    url: source.url.clone(),
                    title: Some(source.title.clone()),
                });
            }
        }

        tracing::info!(
            annotation_count = annotations.len(),
            file_count = generated_files.len(),
            "Built graceful fallback message with file citations"
        );

        // Create a message with empty text but with annotations
        // The frontend will extract file citations and display the images
        output.push(OutputItem::Message(MessageOutput {
            id: message_id(),
            status: OutputStatus::Completed,
            role: OutputRole::Assistant,
            content: vec![OutputContent::OutputText {
                text: String::new(), // Empty text, but annotations carry the images
                annotations,
            }],
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

/// Extract text and images from an MCP CallToolResult.
///
/// Returns the text content and any generated files (images stored in container).
fn extract_content_from_result(
    result: &CallToolResult,
    container_id: &str,
    containers: &ContainerStore,
    file_index_offset: usize,
) -> (String, Vec<GeneratedFile>) {
    use base64::Engine;

    let mut text_parts = Vec::new();
    let mut generated_files = Vec::new();

    for content in &result.content {
        match &content.raw {
            RawContent::Text(tc) => {
                text_parts.push(tc.text.as_str());
            }
            RawContent::Image(img) => {
                // Decode base64 image data
                if let Ok(data) = base64::engine::general_purpose::STANDARD.decode(&img.data) {
                    // Determine file extension from MIME type
                    let ext = match img.mime_type.as_str() {
                        "image/png" => "png",
                        "image/jpeg" => "jpg",
                        "image/gif" => "gif",
                        "image/webp" => "webp",
                        _ => "png",
                    };

                    let index = file_index_offset + generated_files.len();
                    let filename = format!("output_{}.{}", index, ext);

                    // Store in container
                    if let Some(file_id) =
                        containers.add_file(container_id, filename.clone(), data, &img.mime_type)
                    {
                        tracing::info!(
                            file_id = %file_id,
                            filename = %filename,
                            container_id = %container_id,
                            "Stored image from tool execution"
                        );
                        generated_files.push(GeneratedFile {
                            file_id,
                            filename,
                            container_id: container_id.to_string(),
                        });
                    } else {
                        tracing::error!(
                            container_id = %container_id,
                            filename = %filename,
                            "Failed to store image in container"
                        );
                    }
                } else {
                    tracing::warn!("Failed to decode base64 image data");
                }
            }
            _ => {}
        }
    }

    let text = text_parts.join("\n");
    let text = if result.is_error.unwrap_or(false) {
        format!("Error: {}", text)
    } else {
        text
    };

    // Append completion confirmation so the model knows the task is done
    // Explicitly prohibit markdown links (they don't resolve) and extra generations
    let text = if !generated_files.is_empty() {
        format!(
            "{}\n\n[IMAGE GENERATION COMPLETE: The image is now displayed to the user. Do not include markdown image links or references - they will not work. Simply acknowledge the image was created.]",
            text
        )
    } else {
        text
    };

    (text, generated_files)
}

/// Extract WebSearchSource entries from MCP tool result metadata.
///
/// The web search MCP server returns sources in `structuredContent.sources` or
/// as extra fields in the result. We check both locations.
fn extract_sources_from_result(result: &CallToolResult) -> Vec<WebSearchSource> {
    // First try structuredContent which is the official MCP field
    if let Some(structured) = &result.structured_content
        && let Some(sources) = extract_sources_from_value(structured)
    {
        return sources;
    }

    // Also check the _meta field for backwards compatibility
    if let Some(meta) = &result.meta {
        // Meta is a tuple struct wrapping JsonObject (Map<String, Value>)
        let meta_value = Value::Object(meta.0.clone());
        if let Some(sources) = extract_sources_from_value(&meta_value) {
            return sources;
        }
    }

    vec![]
}

/// Extract the query parameter from tool call arguments JSON.
fn extract_query_from_args(args: &str) -> String {
    serde_json::from_str::<Value>(args)
        .ok()
        .and_then(|v| v.get("query")?.as_str().map(String::from))
        .unwrap_or_default()
}

/// Extract sources from a JSON value that may contain a "sources" array.
fn extract_sources_from_value(value: &Value) -> Option<Vec<WebSearchSource>> {
    let sources = value.get("sources")?.as_array()?;

    let result: Vec<WebSearchSource> = sources
        .iter()
        .filter_map(|s| {
            Some(WebSearchSource {
                url: s.get("url")?.as_str()?.to_string(),
                title: s.get("title")?.as_str()?.to_string(),
                snippet: s.get("snippet").and_then(|v| v.as_str()).map(String::from),
            })
        })
        .collect();

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Expand built-in tool types to full function definitions.
async fn expand_tools(mcp: &McpClient, tools: &[Tool]) -> Vec<Tool> {
    let mut expanded = Vec::new();

    for tool in tools {
        match tool {
            Tool::Function(f) => {
                // Pass through function tools as-is
                expanded.push(Tool::Function(f.clone()));
            }
            Tool::Builtin(builtin) => {
                // Expand built-in tool to function definitions from MCP server
                if let Some(mcp_tools) = mcp.get_tools_by_builtin_type(&builtin.tool_type).await {
                    for mcp_tool in mcp_tools {
                        expanded.push(mcp_tool_to_function_tool(mcp_tool));
                    }
                } else {
                    tracing::warn!("Unknown built-in tool type: {}", builtin.tool_type);
                }
            }
        }
    }

    expanded
}

/// Convert an MCP tool to a function tool wrapper.
fn mcp_tool_to_function_tool(mcp_tool: rmcp::model::Tool) -> Tool {
    let parameters = serde_json::to_value(&*mcp_tool.input_schema).ok();

    Tool::Function(FunctionToolWrapper {
        tool_type: "function".to_string(),
        name: mcp_tool.name.to_string(),
        description: mcp_tool.description.map(|d| d.to_string()),
        parameters,
        strict: false,
    })
}

/// Append request input and response output items to a conversation.
///
/// Converts the request input and each `OutputItem` to `ConversationItem` and appends to the store.
fn append_to_conversation(
    store: &InMemoryConversationStore,
    conversation_id: &str,
    req: &CreateResponseRequest,
    output: &[OutputItem],
) {
    use crate::models::{Input, InputItem, MessageContent, MessageInput, Role};
    use crate::translation::{function_call_id, item_id, message_id, reasoning_id};

    let mut all_items: Vec<ConversationItem> = Vec::new();

    // First, append input items from the request
    let input_items: Vec<InputItem> = match &req.input {
        Input::Empty => vec![],
        Input::Text(text) => {
            // Convert simple text to a user message
            vec![InputItem::Message(MessageInput {
                role: Role::User,
                content: MessageContent::Text(text.clone()),
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
    for output_item in output {
        let id = match output_item {
            OutputItem::Message(m) => m.id.clone(),
            OutputItem::FunctionCall(f) => f.id.clone(),
            OutputItem::Reasoning(r) => r.id.clone(),
            OutputItem::WebSearchCall(w) => w.id.clone(),
            _ => item_id(),
        };
        all_items.push(ConversationItem {
            id,
            status: OutputStatus::Completed,
            content: ConversationItemContent::Output(
                serde_json::to_value(output_item).unwrap_or_default(),
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
            "Appended streaming request and response to conversation"
        );
    }
}
