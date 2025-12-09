//! Request translation: Responses API → Chat Completions.

use crate::models::{
    ChatCompletionRequest, ChatContent, ChatContentPart, ChatFunction, ChatFunctionName,
    ChatImageUrl, ChatMessage, ChatRole, ChatTool, ChatToolChoice, ChatToolType, ContentPart,
    CreateResponseRequest, CustomToolCallInput, Input, InputItem, MessageContent, MessageInput,
    ReasoningContentInput, ReasoningInput, Role, Tool, ToolChoice, ToolChoiceMode,
};

/// Translate a Responses API request into a Chat Completions request.
pub fn to_chat_completion(
    req: &CreateResponseRequest,
    previous_messages: Option<Vec<ChatMessage>>,
) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // 1. Add system/instructions if present
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: Some(ChatContent::Text(instructions.clone())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // 2. Add previous conversation context if available
    // When previous_messages is provided, the user input is already included
    // (added by streaming.rs to ensure correct message ordering in tool loops)
    if let Some(prev) = previous_messages {
        messages.extend(prev);
    } else {
        // Fresh request with no conversation history - add input
        messages.extend(input_to_messages(&req.input));
    }

    // 4. Convert tools
    let tools = if req.tools.is_empty() {
        None
    } else {
        let chat_tools: Vec<_> = req.tools.iter().filter_map(tool_to_chat_tool).collect();
        if chat_tools.is_empty() {
            None
        } else {
            Some(chat_tools)
        }
    };

    // 5. Convert tool_choice
    let tool_choice = if tools.is_some() {
        Some(tool_choice_to_chat(&req.tool_choice))
    } else {
        None
    };

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        max_tokens: req.max_output_tokens,
        temperature: Some(req.temperature),
        top_p: Some(req.top_p),
        stream: req.stream,
    }
}

/// Convert Responses API input to Chat Completions messages.
pub fn input_to_messages(input: &Input) -> Vec<ChatMessage> {
    match input {
        Input::Empty => vec![],
        Input::Text(text) => vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text(text.clone())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        Input::Items(items) => items.iter().filter_map(input_item_to_message).collect(),
    }
}

/// Convert a single input item to a Chat message.
fn input_item_to_message(item: &InputItem) -> Option<ChatMessage> {
    match item {
        InputItem::Message(msg) => Some(message_input_to_chat(msg)),
        InputItem::FunctionCallOutput(output) => Some(ChatMessage {
            role: ChatRole::Tool,
            content: Some(ChatContent::Text(output.output.clone())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(output.call_id.clone()),
        }),
        InputItem::FunctionCall(fc) => Some(ChatMessage {
            role: ChatRole::Assistant,
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![crate::models::ChatToolCall {
                id: fc.call_id.clone(),
                tool_type: ChatToolType::Function,
                function: crate::models::ChatFunctionCall {
                    name: fc.name.clone(),
                    arguments: fc.arguments.clone(),
                },
            }]),
            tool_call_id: None,
        }),
        // Custom tool calls are similar to function calls but use free-form text input
        // instead of JSON arguments. We convert them the same way for the chat format.
        InputItem::CustomToolCall(ctc) => Some(custom_tool_call_to_chat(ctc)),
        // Reasoning items are converted to assistant messages with <think> tags
        InputItem::Reasoning(reasoning) => reasoning_input_to_chat(reasoning),
        // Custom tool calls use free-form text input instead of JSON schema.
        // We handle them the same way as function calls, converting to tool messages.
        InputItem::CustomToolCallOutput(output) => Some(ChatMessage {
            role: ChatRole::Tool,
            content: Some(ChatContent::Text(output.output.clone())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(output.call_id.clone()),
        }),
        // Computer use outputs contain screenshots and interaction results.
        // Standard Chat Completions API doesn't support computer use capabilities, so we skip these.
        // A full implementation would convert screenshots to image content parts.
        InputItem::ComputerCallOutput(_) => {
            tracing::warn!("Computer call outputs are not supported; skipping");
            None
        }
        // Item references must be resolved at a higher level where the full item list is available.
        // This function only handles individual items, so references cannot be resolved here.
        InputItem::ItemReference(reference) => {
            tracing::debug!(
                referenced_id = %reference.id,
                "Item reference must be resolved before conversion; skipping"
            );
            None
        }
    }
}

/// Convert CustomToolCallInput to a ChatMessage.
/// Custom tool calls use free-form text input instead of JSON arguments,
/// but we convert them to the same format as function calls for the chat API.
fn custom_tool_call_to_chat(ctc: &CustomToolCallInput) -> ChatMessage {
    ChatMessage {
        role: ChatRole::Assistant,
        content: None,
        reasoning_content: None,
        tool_calls: Some(vec![crate::models::ChatToolCall {
            id: ctc.call_id.clone(),
            tool_type: ChatToolType::Function,
            function: crate::models::ChatFunctionCall {
                name: ctc.name.clone(),
                // Custom tool calls use free-form text, so we pass it as-is
                // The model will interpret it based on the tool definition
                arguments: ctc.input.clone(),
            },
        }]),
        tool_call_id: None,
    }
}

/// Convert ReasoningInput to a ChatMessage with <think> tags.
fn reasoning_input_to_chat(reasoning: &ReasoningInput) -> Option<ChatMessage> {
    // Check for encrypted content
    if reasoning.encrypted_content.is_some() {
        tracing::warn!(
            "Encrypted reasoning content is not yet supported; skipping reasoning input"
        );
        return None;
    }

    // Extract reasoning text from content items
    let reasoning_texts: Vec<String> = reasoning
        .content
        .iter()
        .filter_map(|content_item| match content_item {
            ReasoningContentInput::ReasoningText { text } => Some(text.clone()),
            ReasoningContentInput::Redacted {} => {
                // Skip redacted content
                None
            }
        })
        .collect();

    // If no reasoning text was found, skip this item
    if reasoning_texts.is_empty() {
        return None;
    }

    // Combine all reasoning texts and wrap in <think> tags
    let combined_reasoning = reasoning_texts.join("\n");
    let wrapped_reasoning = format!("<think>{}</think>", combined_reasoning);

    Some(ChatMessage {
        role: ChatRole::Assistant,
        content: Some(ChatContent::Text(wrapped_reasoning)),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    })
}

/// Convert a MessageInput to a ChatMessage.
fn message_input_to_chat(msg: &MessageInput) -> ChatMessage {
    ChatMessage {
        role: role_to_chat_role(msg.role),
        content: Some(message_content_to_chat(&msg.content)),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Convert Responses API role to Chat Completions role.
fn role_to_chat_role(role: Role) -> ChatRole {
    match role {
        Role::System | Role::Developer => ChatRole::System,
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
    }
}

/// Convert message content to Chat format.
fn message_content_to_chat(content: &MessageContent) -> ChatContent {
    match content {
        MessageContent::Text(text) => ChatContent::Text(text.clone()),
        MessageContent::Parts(parts) => {
            ChatContent::Parts(parts.iter().map(content_part_to_chat).collect())
        }
    }
}

/// Convert a content part to Chat format.
fn content_part_to_chat(part: &ContentPart) -> ChatContentPart {
    match part {
        ContentPart::InputText { text } => ChatContentPart::Text { text: text.clone() },
        ContentPart::InputImage { image_url } => ChatContentPart::ImageUrl {
            image_url: ChatImageUrl {
                url: image_url.url.clone(),
                detail: image_url.detail.map(|d| format!("{:?}", d).to_lowercase()),
            },
        },
        ContentPart::InputFile { file } => {
            // Files are typically handled separately; convert to text placeholder
            ChatContentPart::Text {
                text: format!("[File: {}]", file.filename.as_deref().unwrap_or("unknown")),
            }
        }
    }
}

/// Convert a Responses API tool to Chat Completions format.
fn tool_to_chat_tool(tool: &Tool) -> Option<ChatTool> {
    match tool {
        Tool::Function(func) => Some(ChatTool {
            tool_type: ChatToolType::Function,
            function: ChatFunction {
                name: func.name.clone(),
                description: func.description.clone(),
                parameters: func.parameters.clone(),
            },
        }),
        Tool::Builtin(_) => {
            // Built-in tools should be expanded before reaching this point
            // If we get here, the tool wasn't expanded - skip it
            None
        }
    }
}

/// Convert tool choice to Chat Completions format.
fn tool_choice_to_chat(choice: &ToolChoice) -> ChatToolChoice {
    match choice {
        ToolChoice::Mode(mode) => {
            let s = match mode {
                ToolChoiceMode::Auto => "auto",
                ToolChoiceMode::Required => "required",
                ToolChoiceMode::None => "none",
            };
            ChatToolChoice::Mode(s.to_string())
        }
        ToolChoice::Specific(specific) => ChatToolChoice::Specific {
            tool_type: "function".to_string(),
            function: ChatFunctionName {
                name: specific.name.clone(),
            },
        },
    }
}

// ============================================================================
// Image Extraction (for tool injection)
// ============================================================================

/// Strip data URI prefix from base64 image data.
///
/// Converts "data:image/png;base64,abc123" to "abc123".
/// Returns the original string if no base64 data URI prefix is found.
pub fn strip_data_uri_prefix(data: &str) -> &str {
    if let Some(comma_idx) = data.find(',')
        && data[..comma_idx].contains("base64")
    {
        return &data[comma_idx + 1..];
    }
    data
}

/// Attached image with ID and base64 data.
#[derive(Debug, Clone)]
pub struct AttachedImage {
    /// Image identifier (e.g., "image_0")
    pub id: String,
    /// Raw base64 data (data URI prefix stripped)
    pub data: String,
}

/// Extract attached images from request input.
///
/// Returns a list of images found in the input, each with an ID for reference.
/// The ID format is "image_0", "image_1", etc.
pub fn extract_attached_images(input: &Input) -> Vec<AttachedImage> {
    let mut images = Vec::new();
    let mut index = 0;

    match input {
        Input::Empty | Input::Text(_) => {}
        Input::Items(items) => {
            for item in items {
                if let InputItem::Message(msg) = item
                    && let MessageContent::Parts(parts) = &msg.content
                {
                    for part in parts {
                        if let ContentPart::InputImage { image_url } = part {
                            images.push(AttachedImage {
                                id: format!("image_{}", index),
                                data: strip_data_uri_prefix(&image_url.url).to_string(),
                            });
                            index += 1;
                        }
                    }
                }
            }
        }
    }

    images
}

/// Replace images in input with text placeholders for non-vision models.
///
/// Returns a modified Input where images are replaced with descriptive text,
/// allowing the LLM to know images are attached without seeing the data.
pub fn replace_images_with_placeholders(input: &Input) -> Input {
    match input {
        Input::Empty => Input::Empty,
        Input::Text(t) => Input::Text(t.clone()),
        Input::Items(items) => {
            let mut index = 0;
            let modified_items: Vec<InputItem> = items
                .iter()
                .map(|item| match item {
                    InputItem::Message(msg) => {
                        let new_content = match &msg.content {
                            MessageContent::Text(t) => MessageContent::Text(t.clone()),
                            MessageContent::Parts(parts) => {
                                let new_parts: Vec<ContentPart> = parts
                                    .iter()
                                    .map(|part| match part {
                                        ContentPart::InputImage { .. } => {
                                            let placeholder = format!(
                                                "[Attached image: image_{}. Use image_data: \"image_{}\" in tool calls to reference this image.]",
                                                index, index
                                            );
                                            index += 1;
                                            ContentPart::InputText { text: placeholder }
                                        }
                                        other => other.clone(),
                                    })
                                    .collect();
                                MessageContent::Parts(new_parts)
                            }
                        };
                        InputItem::Message(MessageInput {
                            role: msg.role,
                            content: new_content,
                        })
                    }
                    other => other.clone(),
                })
                .collect();
            Input::Items(modified_items)
        }
    }
}

// ============================================================================
// Context Assembly (for previous_response_id chaining)
// ============================================================================

use crate::state::StoredResponse;
use crate::translation::response::response_outputs_to_input_items;

/// Assemble conversation context from a resolved response chain.
///
/// This takes a chain of stored responses (oldest first) and builds the
/// conversation history as `Vec<ChatMessage>` ready for the LLM.
///
/// # Arguments
///
/// * `chain` - Vector of StoredResponse in chronological order (oldest first)
/// * `current_request` - The current request being processed
///
/// # Returns
///
/// A tuple of (resolved_instructions, previous_messages) where:
/// * `resolved_instructions` - System prompt to use (current request takes precedence)
/// * `previous_messages` - Chat messages from the chain (excluding system prompt)
///
/// # Assembly Order
///
/// For a chain [A, B] where A is older:
/// 1. Request A's input → user messages
/// 2. Response A's output → assistant messages
/// 3. Request B's input → user messages
/// 4. Response B's output → assistant messages
///
/// The current request's input is NOT included - that's handled by `to_chat_completion()`.
pub fn assemble_context_from_chain(
    chain: &[StoredResponse],
    current_request: &CreateResponseRequest,
) -> (Option<String>, Vec<ChatMessage>) {
    if chain.is_empty() {
        return (current_request.instructions.clone(), vec![]);
    }

    // Resolve instructions: current request takes precedence
    // If not specified, inherit from the most recent request in the chain that has one
    let resolved_instructions = if current_request.instructions.is_some() {
        current_request.instructions.clone()
    } else {
        // Search chain from newest to oldest for instructions
        chain
            .iter()
            .rev()
            .find_map(|stored| stored.request.instructions.clone())
    };

    let mut messages: Vec<ChatMessage> = Vec::new();

    for stored in chain {
        // 1. Add request input as user/assistant messages
        let request_messages = input_to_messages(&stored.request.input);
        messages.extend(request_messages);

        // 2. Convert response output to input items, then to messages
        let output_as_inputs = response_outputs_to_input_items(&stored.response);
        for input_item in output_as_inputs {
            if let Some(msg) = input_item_to_message(&input_item) {
                messages.push(msg);
            }
        }
    }

    tracing::debug!(
        chain_length = chain.len(),
        message_count = messages.len(),
        has_instructions = resolved_instructions.is_some(),
        "Assembled context from chain"
    );

    (resolved_instructions, messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        CustomToolCallOutputInput, FunctionCallInput, FunctionCallOutputInput, FunctionToolWrapper,
        ReasoningContentInput, ReasoningInput,
    };
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn simple_text_input_converts_to_user_message() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Text("Hello, world!".to_string()),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(chat_req.messages[0].role, ChatRole::User);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text("Hello, world!".to_string()))
        );
    }

    #[test]
    fn instructions_become_system_message() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Text("Hi".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 2);
        assert_eq!(chat_req.messages[0].role, ChatRole::System);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text(
                "You are a helpful assistant.".to_string()
            ))
        );
    }

    #[test]
    fn function_tool_converts_correctly() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Text("What's the weather?".to_string()),
            instructions: None,
            tools: vec![Tool::Function(FunctionToolWrapper {
                tool_type: "function".to_string(),
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
                strict: false,
            })],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert!(chat_req.tools.is_some());
        let tools = chat_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn function_call_output_becomes_tool_message() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Items(vec![
                InputItem::Message(MessageInput {
                    role: Role::User,
                    content: MessageContent::Text("What's the weather?".to_string()),
                }),
                InputItem::FunctionCall(FunctionCallInput {
                    call_id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"location":"Paris"}"#.to_string(),
                    id: None,
                    status: None,
                }),
                InputItem::FunctionCallOutput(FunctionCallOutputInput {
                    call_id: "call_123".to_string(),
                    output: r#"{"temp": 20, "unit": "C"}"#.to_string(),
                    id: None,
                }),
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 3);

        // User message
        assert_eq!(chat_req.messages[0].role, ChatRole::User);

        // Assistant with tool call
        assert_eq!(chat_req.messages[1].role, ChatRole::Assistant);
        assert!(chat_req.messages[1].tool_calls.is_some());

        // Tool result
        assert_eq!(chat_req.messages[2].role, ChatRole::Tool);
        assert_eq!(
            chat_req.messages[2].tool_call_id,
            Some("call_123".to_string())
        );
    }

    #[test]
    fn previous_messages_are_used_directly() {
        // When previous_messages is provided, streaming.rs has already added the
        // current user input to the conversation. So to_chat_completion should NOT
        // add the input again - it just uses the previous_messages as-is.
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Text("Follow up question".to_string()),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        // Simulate what streaming.rs would provide - previous messages WITH current input
        let previous = vec![
            ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Text("First question".to_string())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: Some(ChatContent::Text("First answer".to_string())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Text("Follow up question".to_string())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let chat_req = to_chat_completion(&req, Some(previous));

        // Should have all 3 messages from previous_messages
        assert_eq!(chat_req.messages.len(), 3);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text("First question".to_string()))
        );
        assert_eq!(
            chat_req.messages[2].content,
            Some(ChatContent::Text("Follow up question".to_string()))
        );
    }

    #[test]
    fn reasoning_input_wraps_in_think_tags() {
        let req = CreateResponseRequest {
            model: "gpt-4o".to_string(),
            input: Input::Items(vec![
                InputItem::Reasoning(ReasoningInput {
                    id: Some("reasoning_1".to_string()),
                    content: vec![ReasoningContentInput::ReasoningText {
                        text: "Let me analyze this problem step by step...".to_string(),
                    }],
                    encrypted_content: None,
                }),
                InputItem::Message(MessageInput {
                    role: Role::User,
                    content: MessageContent::Text("Continue the analysis".to_string()),
                }),
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 2);

        // Reasoning becomes assistant message with <think> tags
        assert_eq!(chat_req.messages[0].role, ChatRole::Assistant);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text(
                "<think>Let me analyze this problem step by step...</think>".to_string()
            ))
        );

        // User message follows
        assert_eq!(chat_req.messages[1].role, ChatRole::User);
        assert_eq!(
            chat_req.messages[1].content,
            Some(ChatContent::Text("Continue the analysis".to_string()))
        );
    }

    #[test]
    fn reasoning_with_multiple_text_items_combines() {
        let req = CreateResponseRequest {
            model: "gpt-4o".to_string(),
            input: Input::Items(vec![InputItem::Reasoning(ReasoningInput {
                id: None,
                content: vec![
                    ReasoningContentInput::ReasoningText {
                        text: "First thought".to_string(),
                    },
                    ReasoningContentInput::ReasoningText {
                        text: "Second thought".to_string(),
                    },
                ],
                encrypted_content: None,
            })]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text(
                "<think>First thought\nSecond thought</think>".to_string()
            ))
        );
    }

    #[test]
    fn reasoning_with_redacted_content_skips_redacted() {
        let req = CreateResponseRequest {
            model: "gpt-4o".to_string(),
            input: Input::Items(vec![InputItem::Reasoning(ReasoningInput {
                id: None,
                content: vec![
                    ReasoningContentInput::ReasoningText {
                        text: "Visible thought".to_string(),
                    },
                    ReasoningContentInput::Redacted {},
                    ReasoningContentInput::ReasoningText {
                        text: "Another visible thought".to_string(),
                    },
                ],
                encrypted_content: None,
            })]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(
            chat_req.messages[0].content,
            Some(ChatContent::Text(
                "<think>Visible thought\nAnother visible thought</think>".to_string()
            ))
        );
    }

    #[test]
    fn reasoning_with_only_redacted_content_is_skipped() {
        let req = CreateResponseRequest {
            model: "gpt-4o".to_string(),
            input: Input::Items(vec![
                InputItem::Reasoning(ReasoningInput {
                    id: None,
                    content: vec![ReasoningContentInput::Redacted {}],
                    encrypted_content: None,
                }),
                InputItem::Message(MessageInput {
                    role: Role::User,
                    content: MessageContent::Text("Hello".to_string()),
                }),
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        // Only the user message should be included
        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(chat_req.messages[0].role, ChatRole::User);
    }

    // ========================================================================
    // Context Assembly Tests
    // ========================================================================

    use crate::models::{
        MessageOutput, OutputContent, OutputItem, OutputRole, OutputStatus, Response,
        ResponseStatus, Truncation, Usage,
    };
    use crate::state::StoredResponse;
    use std::time::{Duration, Instant};

    fn make_stored_response(
        id: &str,
        prev_id: Option<&str>,
        input: Input,
        output: Vec<OutputItem>,
        instructions: Option<String>,
    ) -> StoredResponse {
        StoredResponse {
            response: Response {
                id: id.to_string(),
                object: Response::OBJECT,
                created_at: 0,
                status: ResponseStatus::Completed,
                error: None,
                incomplete_details: None,
                instructions: instructions.clone(),
                max_output_tokens: None,
                model: "gpt-4".to_string(),
                output,
                parallel_tool_calls: true,
                previous_response_id: prev_id.map(String::from),
                reasoning: None,
                store: true,
                temperature: 1.0,
                text: None,
                tool_choice: ToolChoice::default(),
                tools: vec![],
                top_p: 1.0,
                truncation: Truncation::default(),
                usage: Usage::default(),
                user: None,
                metadata: serde_json::Value::Null,
            },
            request: CreateResponseRequest {
                model: "gpt-4".to_string(),
                input,
                instructions,
                tools: vec![],
                tool_choice: ToolChoice::default(),
                parallel_tool_calls: true,
                previous_response_id: prev_id.map(String::from),
                max_output_tokens: None,
                max_tool_calls: None,
                temperature: 1.0,
                top_p: 1.0,
                stream: false,
                store: true,
                reasoning: None,
                text: None,
                truncation: Default::default(),
                metadata: None,
                conversation: None,
            },
            created_at: Instant::now(),
            ttl: Duration::from_secs(3600),
        }
    }

    fn make_current_request(input: Input, instructions: Option<String>) -> CreateResponseRequest {
        CreateResponseRequest {
            model: "gpt-4".to_string(),
            input,
            instructions,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        }
    }

    #[test]
    fn empty_chain_returns_empty_messages() {
        let current = make_current_request(Input::Text("Hi".to_string()), None);

        let (instructions, messages) = assemble_context_from_chain(&[], &current);

        assert!(instructions.is_none());
        assert!(messages.is_empty());
    }

    #[test]
    fn empty_chain_preserves_current_instructions() {
        let current = make_current_request(
            Input::Text("Hi".to_string()),
            Some("Be helpful".to_string()),
        );

        let (instructions, messages) = assemble_context_from_chain(&[], &current);

        assert_eq!(instructions, Some("Be helpful".to_string()));
        assert!(messages.is_empty());
    }

    #[test]
    fn single_turn_builds_user_and_assistant_messages() {
        let stored = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![OutputItem::Message(MessageOutput {
                id: "msg_1".to_string(),
                status: OutputStatus::Completed,
                role: OutputRole::Assistant,
                content: vec![OutputContent::OutputText {
                    text: "Hi there!".to_string(),
                    annotations: vec![],
                }],
            })],
            None,
        );
        let chain = vec![stored];
        let current = make_current_request(Input::Text("How are you?".to_string()), None);

        let (_, messages) = assemble_context_from_chain(&chain, &current);

        assert_eq!(messages.len(), 2);
        // First: user input from request A
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(
            messages[0].content,
            Some(ChatContent::Text("Hello".to_string()))
        );
        // Second: assistant output from response A
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(
            messages[1].content,
            Some(ChatContent::Text("Hi there!".to_string()))
        );
    }

    #[test]
    fn two_turn_chain_in_correct_order() {
        let stored_a = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![OutputItem::Message(MessageOutput {
                id: "msg_1".to_string(),
                status: OutputStatus::Completed,
                role: OutputRole::Assistant,
                content: vec![OutputContent::OutputText {
                    text: "Hi!".to_string(),
                    annotations: vec![],
                }],
            })],
            None,
        );
        let stored_b = make_stored_response(
            "resp_b",
            Some("resp_a"),
            Input::Text("What's 2+2?".to_string()),
            vec![OutputItem::Message(MessageOutput {
                id: "msg_2".to_string(),
                status: OutputStatus::Completed,
                role: OutputRole::Assistant,
                content: vec![OutputContent::OutputText {
                    text: "4".to_string(),
                    annotations: vec![],
                }],
            })],
            None,
        );
        let chain = vec![stored_a, stored_b]; // Chronological order
        let current = make_current_request(Input::Text("Thanks!".to_string()), None);

        let (_, messages) = assemble_context_from_chain(&chain, &current);

        assert_eq!(messages.len(), 4);
        // Turn A
        assert_eq!(
            messages[0].content,
            Some(ChatContent::Text("Hello".to_string()))
        );
        assert_eq!(
            messages[1].content,
            Some(ChatContent::Text("Hi!".to_string()))
        );
        // Turn B
        assert_eq!(
            messages[2].content,
            Some(ChatContent::Text("What's 2+2?".to_string()))
        );
        assert_eq!(
            messages[3].content,
            Some(ChatContent::Text("4".to_string()))
        );
    }

    #[test]
    fn instruction_inheritance_from_chain() {
        let stored = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![],
            Some("You are a helpful assistant".to_string()),
        );
        let chain = vec![stored];
        // Current request has no instructions
        let current = make_current_request(Input::Text("Hi".to_string()), None);

        let (instructions, _) = assemble_context_from_chain(&chain, &current);

        assert_eq!(
            instructions,
            Some("You are a helpful assistant".to_string())
        );
    }

    #[test]
    fn current_instructions_take_precedence() {
        let stored = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![],
            Some("Old instructions".to_string()),
        );
        let chain = vec![stored];
        // Current request overrides
        let current = make_current_request(
            Input::Text("Hi".to_string()),
            Some("New instructions".to_string()),
        );

        let (instructions, _) = assemble_context_from_chain(&chain, &current);

        assert_eq!(instructions, Some("New instructions".to_string()));
    }

    #[test]
    fn instruction_inheritance_uses_most_recent() {
        let stored_a = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![],
            Some("Instructions from A".to_string()),
        );
        let stored_b = make_stored_response(
            "resp_b",
            Some("resp_a"),
            Input::Text("Hi".to_string()),
            vec![],
            Some("Instructions from B".to_string()), // More recent
        );
        let chain = vec![stored_a, stored_b];
        let current = make_current_request(Input::Text("Thanks".to_string()), None);

        let (instructions, _) = assemble_context_from_chain(&chain, &current);

        // Should inherit from B (most recent in chain)
        assert_eq!(instructions, Some("Instructions from B".to_string()));
    }

    #[test]
    fn response_with_no_output_still_includes_input() {
        let stored = make_stored_response(
            "resp_a",
            None,
            Input::Text("Hello".to_string()),
            vec![], // No output
            None,
        );
        let chain = vec![stored];
        let current = make_current_request(Input::Text("Hi".to_string()), None);

        let (_, messages) = assemble_context_from_chain(&chain, &current);

        // Should still have the user input
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::User);
    }

    // ========================================================================
    // Image Extraction Tests
    // ========================================================================

    use crate::models::ImageUrl;

    #[test]
    fn strip_data_uri_prefix_removes_base64_prefix() {
        assert_eq!(
            strip_data_uri_prefix("data:image/png;base64,abc123"),
            "abc123"
        );
        assert_eq!(
            strip_data_uri_prefix("data:image/jpeg;base64,xyz789"),
            "xyz789"
        );
    }

    #[test]
    fn strip_data_uri_prefix_preserves_raw_base64() {
        assert_eq!(strip_data_uri_prefix("abc123"), "abc123");
        assert_eq!(
            strip_data_uri_prefix("iVBORw0KGgoAAAANSUhEUg"),
            "iVBORw0KGgoAAAANSUhEUg"
        );
    }

    #[test]
    fn strip_data_uri_prefix_preserves_non_base64_data_uri() {
        // Only strip if it's base64 encoded
        assert_eq!(
            strip_data_uri_prefix("data:text/plain,hello"),
            "data:text/plain,hello"
        );
    }

    #[test]
    fn extract_attached_images_finds_images_in_parts() {
        let input = Input::Items(vec![InputItem::Message(MessageInput {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::InputText {
                    text: "Transform this".to_string(),
                },
                ContentPart::InputImage {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc123".to_string(),
                        detail: None,
                    },
                },
            ]),
        })]);

        let images = extract_attached_images(&input);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].id, "image_0");
        assert_eq!(images[0].data, "abc123"); // Data URI prefix stripped
    }

    #[test]
    fn extract_attached_images_handles_multiple_images() {
        let input = Input::Items(vec![InputItem::Message(MessageInput {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::InputImage {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,first".to_string(),
                        detail: None,
                    },
                },
                ContentPart::InputImage {
                    image_url: ImageUrl {
                        url: "data:image/jpeg;base64,second".to_string(),
                        detail: None,
                    },
                },
            ]),
        })]);

        let images = extract_attached_images(&input);
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].id, "image_0");
        assert_eq!(images[0].data, "first");
        assert_eq!(images[1].id, "image_1");
        assert_eq!(images[1].data, "second");
    }

    #[test]
    fn extract_attached_images_empty_for_text_only() {
        let input = Input::Text("Hello".to_string());
        let images = extract_attached_images(&input);
        assert!(images.is_empty());
    }

    #[test]
    fn extract_attached_images_empty_for_empty_input() {
        let input = Input::Empty;
        let images = extract_attached_images(&input);
        assert!(images.is_empty());
    }

    #[test]
    fn replace_images_with_placeholders_preserves_text() {
        let input = Input::Items(vec![InputItem::Message(MessageInput {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::InputText {
                    text: "Transform this".to_string(),
                },
                ContentPart::InputImage {
                    image_url: ImageUrl {
                        url: "base64data".to_string(),
                        detail: None,
                    },
                },
            ]),
        })]);

        let result = replace_images_with_placeholders(&input);

        if let Input::Items(items) = result {
            if let InputItem::Message(msg) = &items[0] {
                if let MessageContent::Parts(parts) = &msg.content {
                    assert_eq!(parts.len(), 2);
                    // First part should be text unchanged
                    if let ContentPart::InputText { text } = &parts[0] {
                        assert_eq!(text, "Transform this");
                    } else {
                        panic!("Expected InputText for first part");
                    }
                    // Second part should be placeholder
                    if let ContentPart::InputText { text } = &parts[1] {
                        assert!(text.contains("image_0"));
                        assert!(text.contains("Attached image"));
                    } else {
                        panic!("Expected InputText placeholder for second part");
                    }
                } else {
                    panic!("Expected Parts content");
                }
            } else {
                panic!("Expected Message item");
            }
        } else {
            panic!("Expected Items input");
        }
    }

    #[test]
    fn replace_images_with_placeholders_text_input_unchanged() {
        let input = Input::Text("Hello world".to_string());
        let result = replace_images_with_placeholders(&input);

        if let Input::Text(text) = result {
            assert_eq!(text, "Hello world");
        } else {
            panic!("Expected Text input");
        }
    }

    // ========================================================================
    // Custom Tool Call Input Tests
    // ========================================================================

    use crate::models::CustomToolCallInput;

    #[test]
    fn custom_tool_call_input_converts_to_assistant_message() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Items(vec![
                InputItem::Message(MessageInput {
                    role: Role::User,
                    content: MessageContent::Text("Help me analyze this".to_string()),
                }),
                InputItem::CustomToolCall(CustomToolCallInput {
                    call_id: "ctc_123".to_string(),
                    name: "my_custom_tool".to_string(),
                    input: "free form text input".to_string(),
                    id: Some("item_456".to_string()),
                    status: None,
                }),
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 2);

        // First: user message
        assert_eq!(chat_req.messages[0].role, ChatRole::User);

        // Second: assistant with tool call (custom tool call)
        assert_eq!(chat_req.messages[1].role, ChatRole::Assistant);
        assert!(chat_req.messages[1].tool_calls.is_some());
        let tool_calls = chat_req.messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "ctc_123");
        assert_eq!(tool_calls[0].function.name, "my_custom_tool");
        assert_eq!(tool_calls[0].function.arguments, "free form text input");
    }

    #[test]
    fn custom_tool_call_followed_by_output_converts_correctly() {
        let req = CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: Input::Items(vec![
                InputItem::CustomToolCall(CustomToolCallInput {
                    call_id: "ctc_789".to_string(),
                    name: "analyzer".to_string(),
                    input: "analyze the data".to_string(),
                    id: None,
                    status: None,
                }),
                InputItem::CustomToolCallOutput(CustomToolCallOutputInput {
                    call_id: "ctc_789".to_string(),
                    output: "Analysis complete: all systems normal".to_string(),
                    id: None,
                }),
            ]),
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: true,
            previous_response_id: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            reasoning: None,
            text: None,
            truncation: Default::default(),
            metadata: None,
            conversation: None,
        };

        let chat_req = to_chat_completion(&req, None);

        assert_eq!(chat_req.messages.len(), 2);

        // First: assistant with custom tool call
        assert_eq!(chat_req.messages[0].role, ChatRole::Assistant);
        assert!(chat_req.messages[0].tool_calls.is_some());

        // Second: tool result
        assert_eq!(chat_req.messages[1].role, ChatRole::Tool);
        assert_eq!(
            chat_req.messages[1].tool_call_id,
            Some("ctc_789".to_string())
        );
        assert_eq!(
            chat_req.messages[1].content,
            Some(ChatContent::Text(
                "Analysis complete: all systems normal".to_string()
            ))
        );
    }
}
