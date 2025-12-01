//! Response translation: Chat Completions → Responses API.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::execution::GeneratedFile;
use crate::models::{
    Annotation, ChatCompletionResponse, ChatContent, ChatMessage, ChatRole, CreateResponseRequest,
    FunctionCallOutput as FunctionCallOutputItem, InputTokensDetails, MessageOutput, OutputContent,
    OutputItem, OutputRole, OutputStatus, OutputTokensDetails, ReasoningContent, ReasoningOutput,
    Response, ResponseStatus, Usage,
};

use super::ids::{function_call_id, message_id, reasoning_id, response_id};

/// Build a Response object from a Chat Completion response.
pub fn from_chat_completion(
    chat_resp: &ChatCompletionResponse,
    req: &CreateResponseRequest,
    generated_files: Vec<GeneratedFile>,
) -> Response {
    Response {
        id: response_id(),
        object: Response::OBJECT,
        created_at: unix_timestamp(),
        status: ResponseStatus::Completed,
        error: None,
        incomplete_details: None,
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        model: chat_resp.model.clone(),
        output: extract_output_items(chat_resp, &generated_files),
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
        usage: extract_usage(chat_resp),
        user: None,
        metadata: req.metadata.clone().unwrap_or(Value::Null),
    }
}

/// Parse reasoning tags from text and return (reasoning_text, remaining_text).
/// Returns (None, original_text) if no <think> tags found.
fn parse_reasoning_tags(text: &str) -> (Option<String>, String) {
    // Look for <think>...</think> tags
    if let Some(start_idx) = text.find("<think>")
        && let Some(end_idx) = text.find("</think>")
    {
        let think_start = start_idx + "<think>".len();
        let reasoning_text = text[think_start..end_idx].to_string();

        // Remove the <think>...</think> portion from the original text
        let before = &text[..start_idx];
        let after = &text[end_idx + "</think>".len()..];
        let remaining = format!("{}{}", before, after).trim().to_string();

        return (Some(reasoning_text), remaining);
    }

    (None, text.to_string())
}

/// Extract output items from Chat Completion response.
///
/// This extracts output items that can be directly produced by Chat Completions:
/// - FunctionCall: Converted from tool_calls in the response
/// - Message: Converted from text content
/// - Reasoning: Extracted from <think> tags in content
///
/// Other OutputItem variants (CustomToolCall, WebSearchCall, FileSearchCall,
/// CodeInterpreterCall, ComputerCall) are specialized types that would be
/// produced by external tool systems, not directly by the inference backend.
fn extract_output_items(
    chat_resp: &ChatCompletionResponse,
    generated_files: &[GeneratedFile],
) -> Vec<OutputItem> {
    let mut items = Vec::new();

    for choice in &chat_resp.choices {
        // Handle tool calls
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                items.push(OutputItem::FunctionCall(FunctionCallOutputItem {
                    id: function_call_id(),
                    call_id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                    status: OutputStatus::Completed,
                }));
            }
        }

        // Handle text content
        if let Some(content) = &choice.message.content {
            let text = match content {
                ChatContent::Text(t) => t.clone(),
                ChatContent::Parts(parts) => {
                    // Concatenate text parts
                    parts
                        .iter()
                        .filter_map(|p| match p {
                            crate::models::ChatContentPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("")
                }
            };

            if !text.is_empty() {
                // Parse reasoning tags if present
                let (reasoning, remaining_text) = parse_reasoning_tags(&text);

                // If we found reasoning, emit it first
                if let Some(reasoning_text) = reasoning {
                    items.push(OutputItem::Reasoning(ReasoningOutput {
                        id: reasoning_id(),
                        status: OutputStatus::Completed,
                        content: vec![ReasoningContent::ReasoningText {
                            text: reasoning_text,
                        }],
                        summary: vec![],
                        encrypted_content: None,
                    }));
                }

                // Build annotations for generated files
                let annotations: Vec<Annotation> = generated_files
                    .iter()
                    .map(|f| Annotation::ContainerFileCitation {
                        container_id: f.container_id.clone(),
                        file_id: f.file_id.clone(),
                        filename: f.filename.clone(),
                    })
                    .collect();

                // Then emit the message if there's remaining text
                if !remaining_text.is_empty() {
                    items.push(OutputItem::Message(MessageOutput {
                        id: message_id(),
                        status: OutputStatus::Completed,
                        role: OutputRole::Assistant,
                        content: vec![OutputContent::OutputText {
                            text: remaining_text,
                            annotations,
                        }],
                    }));
                }
            }
        }
    }

    items
}

/// Extract usage statistics.
fn extract_usage(chat_resp: &ChatCompletionResponse) -> Usage {
    chat_resp
        .usage
        .as_ref()
        .map(|u| Usage {
            input_tokens: u.prompt_tokens,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 0 }),
            output_tokens: u.completion_tokens,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: u.total_tokens,
        })
        .unwrap_or_default()
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_secs() as i64
}

/// Check if response contains tool calls that need execution.
pub fn has_pending_tool_calls(chat_resp: &ChatCompletionResponse) -> bool {
    chat_resp.choices.iter().any(|c| {
        c.message
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty())
    })
}

/// Extract tool calls from a Chat Completion response for execution.
pub fn extract_tool_calls(chat_resp: &ChatCompletionResponse) -> Vec<PendingToolCall> {
    let mut calls = Vec::new();

    for choice in &chat_resp.choices {
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tc in tool_calls {
                calls.push(PendingToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                });
            }
        }
    }

    calls
}

/// A tool call that needs to be executed.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Convert assistant message with tool calls to ChatMessage for context.
pub fn assistant_tool_call_message(chat_resp: &ChatCompletionResponse) -> Option<ChatMessage> {
    for choice in &chat_resp.choices {
        if choice.message.tool_calls.is_some() {
            return Some(choice.message.clone());
        }
    }
    None
}

/// Create a tool result message.
pub fn tool_result_message(call_id: String, result: String) -> ChatMessage {
    ChatMessage {
        role: ChatRole::Tool,
        content: Some(ChatContent::Text(result)),
        tool_calls: None,
        tool_call_id: Some(call_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        ChatChoice, ChatFunctionCall, ChatToolCall, ChatToolType, ChatUsage, FinishReason,
        ToolChoice, Truncation,
    };
    use pretty_assertions::assert_eq;

    fn make_request() -> CreateResponseRequest {
        CreateResponseRequest {
            model: "gpt-4".to_string(),
            input: crate::models::Input::Text("Hello".to_string()),
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
            truncation: Truncation::default(),
            metadata: None,
        }
    }

    #[test]
    fn text_response_converts_to_message_output() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text("Hello! How can I help?".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        assert_eq!(resp.status, ResponseStatus::Completed);
        assert_eq!(resp.output.len(), 1);

        match &resp.output[0] {
            OutputItem::Message(msg) => {
                assert_eq!(msg.role, OutputRole::Assistant);
                assert_eq!(msg.status, OutputStatus::Completed);
                match &msg.content[0] {
                    OutputContent::OutputText { text, .. } => {
                        assert_eq!(text, "Hello! How can I help?");
                    }
                    _ => panic!("expected OutputText"),
                }
            }
            _ => panic!("expected Message output"),
        }
    }

    #[test]
    fn tool_call_response_converts_to_function_call_output() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_abc123".to_string(),
                        tool_type: ChatToolType::Function,
                        function: ChatFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location":"Paris"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
            }),
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        assert_eq!(resp.output.len(), 1);

        match &resp.output[0] {
            OutputItem::FunctionCall(fc) => {
                assert_eq!(fc.call_id, "call_abc123");
                assert_eq!(fc.name, "get_weather");
                assert_eq!(fc.arguments, r#"{"location":"Paris"}"#);
                assert_eq!(fc.status, OutputStatus::Completed);
            }
            _ => panic!("expected FunctionCall output"),
        }
    }

    #[test]
    fn multiple_tool_calls_all_extracted() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: None,
                    tool_calls: Some(vec![
                        ChatToolCall {
                            id: "call_1".to_string(),
                            tool_type: ChatToolType::Function,
                            function: ChatFunctionCall {
                                name: "get_weather".to_string(),
                                arguments: r#"{"location":"Paris"}"#.to_string(),
                            },
                        },
                        ChatToolCall {
                            id: "call_2".to_string(),
                            tool_type: ChatToolType::Function,
                            function: ChatFunctionCall {
                                name: "get_weather".to_string(),
                                arguments: r#"{"location":"London"}"#.to_string(),
                            },
                        },
                    ]),
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: None,
        };

        let pending = extract_tool_calls(&chat_resp);

        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, "call_1");
        assert_eq!(pending[1].id, "call_2");
    }

    #[test]
    fn usage_is_correctly_mapped() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text("Hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.output_tokens, 50);
        assert_eq!(resp.usage.total_tokens, 150);
    }

    #[test]
    fn response_id_has_correct_format() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![],
            usage: None,
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        assert!(resp.id.starts_with("resp_"), "got: {}", resp.id);
    }

    #[test]
    fn has_pending_tool_calls_detects_correctly() {
        let with_tools = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_1".to_string(),
                        tool_type: ChatToolType::Function,
                        function: ChatFunctionCall {
                            name: "test".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: None,
        };

        let without_tools = ChatCompletionResponse {
            id: "2".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text("Hello".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: None,
        };

        assert!(has_pending_tool_calls(&with_tools));
        assert!(!has_pending_tool_calls(&without_tools));
    }

    #[test]
    fn reasoning_tags_are_parsed_correctly() {
        let (reasoning, remaining) =
            parse_reasoning_tags("<think>Let me think step by step...</think>The answer is 42.");

        assert_eq!(reasoning, Some("Let me think step by step...".to_string()));
        assert_eq!(remaining, "The answer is 42.");
    }

    #[test]
    fn no_reasoning_tags_returns_original_text() {
        let (reasoning, remaining) = parse_reasoning_tags("Just a normal response.");

        assert_eq!(reasoning, None);
        assert_eq!(remaining, "Just a normal response.");
    }

    #[test]
    fn reasoning_response_creates_both_reasoning_and_message_output() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "deepseek-r1".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text(
                        "<think>Let me analyze this problem step by step...</think>The answer is 42."
                            .to_string(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: None,
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        assert_eq!(resp.output.len(), 2);

        // First item should be reasoning
        match &resp.output[0] {
            OutputItem::Reasoning(reasoning) => {
                assert_eq!(reasoning.status, OutputStatus::Completed);
                assert_eq!(reasoning.content.len(), 1);
                match &reasoning.content[0] {
                    ReasoningContent::ReasoningText { text } => {
                        assert_eq!(text, "Let me analyze this problem step by step...");
                    }
                    _ => panic!("expected ReasoningText"),
                }
            }
            _ => panic!("expected Reasoning output"),
        }

        // Second item should be message
        match &resp.output[1] {
            OutputItem::Message(msg) => {
                assert_eq!(msg.role, OutputRole::Assistant);
                assert_eq!(msg.status, OutputStatus::Completed);
                match &msg.content[0] {
                    OutputContent::OutputText { text, .. } => {
                        assert_eq!(text, "The answer is 42.");
                    }
                    _ => panic!("expected OutputText"),
                }
            }
            _ => panic!("expected Message output"),
        }
    }

    #[test]
    fn reasoning_only_response_creates_only_reasoning_output() {
        let chat_resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "deepseek-r1".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text(
                        "<think>Just thinking, no response</think>".to_string(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: None,
        };

        let resp = from_chat_completion(&chat_resp, &make_request(), vec![]);

        // Should only have reasoning output, no message
        assert_eq!(resp.output.len(), 1);

        match &resp.output[0] {
            OutputItem::Reasoning(reasoning) => match &reasoning.content[0] {
                ReasoningContent::ReasoningText { text } => {
                    assert_eq!(text, "Just thinking, no response");
                }
                _ => panic!("expected ReasoningText"),
            },
            _ => panic!("expected Reasoning output"),
        }
    }
}
