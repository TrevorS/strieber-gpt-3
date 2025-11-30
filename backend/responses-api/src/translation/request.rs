//! Request translation: Responses API → Chat Completions.

use crate::models::{
    ChatCompletionRequest, ChatContent, ChatContentPart, ChatFunction, ChatFunctionName,
    ChatImageUrl, ChatMessage, ChatRole, ChatTool, ChatToolChoice, ChatToolType, ContentPart,
    CreateResponseRequest, Input, InputItem, MessageContent, MessageInput, ReasoningContentInput,
    ReasoningInput, Role, Tool, ToolChoice, ToolChoiceMode,
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
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // 2. Add previous conversation context if available
    if let Some(prev) = previous_messages {
        messages.extend(prev);
    }

    // 3. Convert input to messages
    messages.extend(input_to_messages(&req.input));

    // 4. Convert tools
    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(req.tools.iter().map(tool_to_chat_tool).collect())
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
fn input_to_messages(input: &Input) -> Vec<ChatMessage> {
    match input {
        Input::Empty => vec![],
        Input::Text(text) => vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Text(text.clone())),
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
            tool_calls: None,
            tool_call_id: Some(output.call_id.clone()),
        }),
        InputItem::FunctionCall(fc) => Some(ChatMessage {
            role: ChatRole::Assistant,
            content: None,
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
        // Reasoning items are converted to assistant messages with <think> tags
        InputItem::Reasoning(reasoning) => reasoning_input_to_chat(reasoning),
        // Custom tool calls use free-form text input instead of JSON schema.
        // We handle them the same way as function calls, converting to tool messages.
        InputItem::CustomToolCallOutput(output) => Some(ChatMessage {
            role: ChatRole::Tool,
            content: Some(ChatContent::Text(output.output.clone())),
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
        tool_calls: None,
        tool_call_id: None,
    })
}

/// Convert a MessageInput to a ChatMessage.
fn message_input_to_chat(msg: &MessageInput) -> ChatMessage {
    ChatMessage {
        role: role_to_chat_role(msg.role),
        content: Some(message_content_to_chat(&msg.content)),
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
fn tool_to_chat_tool(tool: &Tool) -> ChatTool {
    match tool {
        Tool::Function(func) => ChatTool {
            tool_type: ChatToolType::Function,
            function: ChatFunction {
                name: func.name.clone(),
                description: func.description.clone(),
                parameters: func.parameters.clone(),
            },
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        FunctionCallInput, FunctionCallOutputInput, FunctionTool, ReasoningContentInput,
        ReasoningInput,
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
            tools: vec![Tool::Function(FunctionTool {
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
    fn previous_messages_are_prepended() {
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
        };

        let previous = vec![
            ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Text("First question".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: Some(ChatContent::Text("First answer".to_string())),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let chat_req = to_chat_completion(&req, Some(previous));

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
        };

        let chat_req = to_chat_completion(&req, None);

        // Only the user message should be included
        assert_eq!(chat_req.messages.len(), 1);
        assert_eq!(chat_req.messages[0].role, ChatRole::User);
    }
}
