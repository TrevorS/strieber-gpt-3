//! Response translation: Chat Completions → Responses API.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::execution::GeneratedFile;
use crate::models::{
    Annotation, ChatCompletionResponse, ChatContent, ChatMessage, ChatRole, CreateResponseRequest,
    CustomToolCallOutputInput, FunctionCallInput, FunctionCallOutput as FunctionCallOutputItem,
    InputItem, InputTokensDetails, MessageInput, MessageOutput, OutputContent, OutputItem,
    OutputRole, OutputStatus, OutputTokensDetails, ReasoningContent, ReasoningContentInput,
    ReasoningInput, ReasoningOutput, Response, ResponseStatus, Role, Usage,
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
pub fn parse_reasoning_tags(text: &str) -> (Option<String>, String) {
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

// ============================================================================
// Citation Parsing (for web search sources)
// ============================================================================

use crate::models::WebSearchSource;

/// A parsed citation marker from text (e.g., "[1]", "[2]").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CitationMarker {
    /// 1-based citation number
    pub number: u32,
    /// Start position in text (byte index)
    pub start: usize,
    /// End position in text (byte index)
    pub end: usize,
}

/// Parse citation markers like [1], [2], etc. from text.
/// Returns a list of markers with their positions.
pub fn parse_citation_markers(text: &str) -> Vec<CitationMarker> {
    let mut markers = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Look for a closing bracket
            let start = i;
            i += 1;

            // Parse digits
            let digit_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }

            // Check if we found digits followed by ]
            if i > digit_start && i < bytes.len() && bytes[i] == b']' {
                let digit_str = &text[digit_start..i];
                if let Ok(num) = digit_str.parse::<u32>()
                    && num > 0
                {
                    markers.push(CitationMarker {
                        number: num,
                        start,
                        end: i + 1,
                    });
                }
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    markers
}

/// Build URL citation annotations from text and web search sources.
/// Maps [1], [2], etc. markers to their corresponding sources.
pub fn build_url_citations(text: &str, sources: &[WebSearchSource]) -> Vec<Annotation> {
    let markers = parse_citation_markers(text);

    markers
        .iter()
        .filter_map(|marker| {
            // Citation numbers are 1-based, sources are 0-indexed
            let source_idx = (marker.number - 1) as usize;
            sources
                .get(source_idx)
                .map(|source| Annotation::UrlCitation {
                    url: source.url.clone(),
                    title: Some(source.title.clone()),
                    start_index: marker.start as u32,
                    end_index: marker.end as u32,
                })
        })
        .collect()
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
                    output: None,
                }));
            }
        }

        // Check for reasoning_content field first (gpt-oss, DeepSeek-R1, etc.)
        // llama.cpp sends this as a separate field when using `--reasoning-format auto`
        if let Some(reasoning_text) = &choice.message.reasoning_content
            && !reasoning_text.is_empty()
        {
            items.push(OutputItem::Reasoning(ReasoningOutput {
                id: reasoning_id(),
                status: OutputStatus::Completed,
                content: vec![ReasoningContent::ReasoningText {
                    text: reasoning_text.clone(),
                }],
                summary: vec![],
                encrypted_content: None,
            }));
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
                // If no reasoning_content field was present, try parsing <think> tags
                // (fallback for Qwen-QwQ and other models that use tags)
                let (reasoning, remaining_text) = if choice.message.reasoning_content.is_none() {
                    parse_reasoning_tags(&text)
                } else {
                    // reasoning_content field was present, don't parse tags
                    (None, text.clone())
                };

                // If we found reasoning from tags, emit it
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
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: Some(call_id),
    }
}

// ============================================================================
// Output-to-Input Conversion (for previous_response_id chaining)
// ============================================================================

/// Convert a single output item to an input item for context reconstruction.
///
/// Returns `None` for built-in tool calls (web_search, file_search, code_interpreter, computer)
/// since their results are already incorporated into the conversation context.
pub fn output_item_to_input_item(item: &OutputItem) -> Option<InputItem> {
    match item {
        OutputItem::Message(msg) => Some(InputItem::Message(MessageInput {
            role: output_role_to_input_role(msg.role),
            content: output_content_to_message_content(&msg.content),
        })),
        OutputItem::Reasoning(reasoning) => Some(InputItem::Reasoning(ReasoningInput {
            id: Some(reasoning.id.clone()),
            content: reasoning
                .content
                .iter()
                .map(reasoning_content_to_input)
                .collect(),
            encrypted_content: reasoning.encrypted_content.clone(),
        })),
        OutputItem::FunctionCall(fc) => Some(InputItem::FunctionCall(FunctionCallInput {
            call_id: fc.call_id.clone(),
            name: fc.name.clone(),
            arguments: fc.arguments.clone(),
            id: Some(fc.id.clone()),
            status: Some(format!("{:?}", fc.status).to_lowercase()),
        })),
        OutputItem::CustomToolCall(ctc) => {
            // Custom tool calls convert to CustomToolCallOutput with the input as output
            // This preserves the tool call in context
            Some(InputItem::CustomToolCallOutput(CustomToolCallOutputInput {
                call_id: ctc.call_id.clone(),
                output: ctc.input.clone(),
                id: Some(ctc.id.clone()),
            }))
        }
        // Built-in tools are skipped - their results are already in the context
        OutputItem::WebSearchCall(_) => None,
        OutputItem::FileSearchCall(_) => None,
        OutputItem::CodeInterpreterCall(_) => None,
        OutputItem::ComputerCall(_) => None,
    }
}

/// Convert output role to input role.
fn output_role_to_input_role(role: OutputRole) -> Role {
    match role {
        OutputRole::Assistant => Role::Assistant,
    }
}

/// Convert output content to message content for input.
fn output_content_to_message_content(content: &[OutputContent]) -> crate::models::MessageContent {
    // If there's a single text item, return it as Text
    if let [OutputContent::OutputText { text, .. }] = content {
        return crate::models::MessageContent::Text(text.clone());
    }

    // For multiple parts or non-text content, build Parts
    let parts: Vec<crate::models::ContentPart> = content
        .iter()
        .map(|c| match c {
            OutputContent::OutputText { text, .. } => {
                crate::models::ContentPart::InputText { text: text.clone() }
            }
            OutputContent::Refusal { refusal } => crate::models::ContentPart::InputText {
                text: format!("[Refusal: {}]", refusal),
            },
        })
        .collect();

    crate::models::MessageContent::Parts(parts)
}

/// Convert reasoning content from output to input format.
fn reasoning_content_to_input(content: &ReasoningContent) -> ReasoningContentInput {
    match content {
        ReasoningContent::ReasoningText { text } => {
            ReasoningContentInput::ReasoningText { text: text.clone() }
        }
        ReasoningContent::Redacted {} => ReasoningContentInput::Redacted {},
    }
}

/// Convert all output items from a response to input items.
///
/// This is used when reconstructing conversation context from previous responses.
/// Built-in tool calls are filtered out as their results are already in context.
pub fn response_outputs_to_input_items(response: &Response) -> Vec<InputItem> {
    response
        .output
        .iter()
        .filter_map(output_item_to_input_item)
        .collect()
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
            conversation: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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
                    reasoning_content: None,
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

    // ========================================================================
    // Output-to-Input Conversion Tests
    // ========================================================================

    use crate::models::{
        CodeInterpreterCallOutput, CustomToolCallOutput, FileSearchCallOutput, MessageContent,
        WebSearchCallOutput,
    };

    #[test]
    fn message_output_converts_to_message_input() {
        let output = OutputItem::Message(MessageOutput {
            id: "msg_123".to_string(),
            status: OutputStatus::Completed,
            role: OutputRole::Assistant,
            content: vec![OutputContent::OutputText {
                text: "Hello, world!".to_string(),
                annotations: vec![],
            }],
        });

        let input = output_item_to_input_item(&output).expect("should convert");

        match input {
            InputItem::Message(msg) => {
                assert_eq!(msg.role, Role::Assistant);
                match msg.content {
                    MessageContent::Text(text) => assert_eq!(text, "Hello, world!"),
                    _ => panic!("expected Text content"),
                }
            }
            _ => panic!("expected Message input"),
        }
    }

    #[test]
    fn reasoning_output_converts_to_reasoning_input() {
        let output = OutputItem::Reasoning(ReasoningOutput {
            id: "reasoning_123".to_string(),
            status: OutputStatus::Completed,
            content: vec![ReasoningContent::ReasoningText {
                text: "Let me think...".to_string(),
            }],
            summary: vec![],
            encrypted_content: None,
        });

        let input = output_item_to_input_item(&output).expect("should convert");

        match input {
            InputItem::Reasoning(reasoning) => {
                assert_eq!(reasoning.id, Some("reasoning_123".to_string()));
                assert_eq!(reasoning.content.len(), 1);
                match &reasoning.content[0] {
                    ReasoningContentInput::ReasoningText { text } => {
                        assert_eq!(text, "Let me think...");
                    }
                    _ => panic!("expected ReasoningText"),
                }
            }
            _ => panic!("expected Reasoning input"),
        }
    }

    #[test]
    fn function_call_output_converts_to_function_call_input() {
        let output = OutputItem::FunctionCall(FunctionCallOutputItem {
            id: "fc_123".to_string(),
            call_id: "call_abc".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris"}"#.to_string(),
            status: OutputStatus::Completed,
            output: None,
        });

        let input = output_item_to_input_item(&output).expect("should convert");

        match input {
            InputItem::FunctionCall(fc) => {
                assert_eq!(fc.call_id, "call_abc");
                assert_eq!(fc.name, "get_weather");
                assert_eq!(fc.arguments, r#"{"location":"Paris"}"#);
                assert_eq!(fc.id, Some("fc_123".to_string()));
            }
            _ => panic!("expected FunctionCall input"),
        }
    }

    #[test]
    fn custom_tool_call_converts_to_custom_tool_call_output_input() {
        let output = OutputItem::CustomToolCall(CustomToolCallOutput {
            id: "ctc_123".to_string(),
            call_id: "call_xyz".to_string(),
            name: "custom_tool".to_string(),
            input: "some free form input".to_string(),
            status: OutputStatus::Completed,
        });

        let input = output_item_to_input_item(&output).expect("should convert");

        match input {
            InputItem::CustomToolCallOutput(ctc) => {
                assert_eq!(ctc.call_id, "call_xyz");
                assert_eq!(ctc.output, "some free form input");
                assert_eq!(ctc.id, Some("ctc_123".to_string()));
            }
            _ => panic!("expected CustomToolCallOutput input"),
        }
    }

    #[test]
    fn web_search_call_returns_none() {
        let output = OutputItem::WebSearchCall(WebSearchCallOutput {
            id: "ws_123".to_string(),
            status: OutputStatus::Completed,
            action: None,
        });

        assert!(output_item_to_input_item(&output).is_none());
    }

    #[test]
    fn file_search_call_returns_none() {
        let output = OutputItem::FileSearchCall(FileSearchCallOutput {
            id: "fs_123".to_string(),
            status: OutputStatus::Completed,
            results: vec![],
        });

        assert!(output_item_to_input_item(&output).is_none());
    }

    #[test]
    fn code_interpreter_call_returns_none() {
        let output = OutputItem::CodeInterpreterCall(CodeInterpreterCallOutput {
            id: "ci_123".to_string(),
            status: OutputStatus::Completed,
            code: None,
            outputs: vec![],
        });

        assert!(output_item_to_input_item(&output).is_none());
    }

    #[test]
    fn response_outputs_to_input_items_filters_builtin_tools() {
        let response = Response {
            id: "resp_123".to_string(),
            object: Response::OBJECT,
            created_at: 0,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-4".to_string(),
            output: vec![
                OutputItem::Reasoning(ReasoningOutput {
                    id: "r_1".to_string(),
                    status: OutputStatus::Completed,
                    content: vec![ReasoningContent::ReasoningText {
                        text: "thinking".to_string(),
                    }],
                    summary: vec![],
                    encrypted_content: None,
                }),
                OutputItem::WebSearchCall(WebSearchCallOutput {
                    id: "ws_1".to_string(),
                    status: OutputStatus::Completed,
                    action: None,
                }),
                OutputItem::Message(MessageOutput {
                    id: "m_1".to_string(),
                    status: OutputStatus::Completed,
                    role: OutputRole::Assistant,
                    content: vec![OutputContent::OutputText {
                        text: "response".to_string(),
                        annotations: vec![],
                    }],
                }),
            ],
            parallel_tool_calls: true,
            previous_response_id: None,
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
        };

        let inputs = response_outputs_to_input_items(&response);

        // Should have 2 items (reasoning + message), not the web search
        assert_eq!(inputs.len(), 2);

        // First should be reasoning
        assert!(matches!(inputs[0], InputItem::Reasoning(_)));

        // Second should be message
        assert!(matches!(inputs[1], InputItem::Message(_)));
    }

    // ========================================================================
    // Citation Parsing Tests
    // ========================================================================

    use super::{build_url_citations, parse_citation_markers};

    #[test]
    fn parse_citation_markers_finds_single_marker() {
        let markers = parse_citation_markers("According to [1], the answer is 42.");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].number, 1);
        assert_eq!(markers[0].start, 13); // Position of '['
        assert_eq!(markers[0].end, 16); // Position after ']'
    }

    #[test]
    fn parse_citation_markers_finds_multiple_markers() {
        let markers = parse_citation_markers("Sources [1] and [2] confirm this [3].");
        assert_eq!(markers.len(), 3);
        assert_eq!(markers[0].number, 1);
        assert_eq!(markers[1].number, 2);
        assert_eq!(markers[2].number, 3);
    }

    #[test]
    fn parse_citation_markers_ignores_non_numeric_brackets() {
        let markers = parse_citation_markers("See [note] and [here] for details [1].");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].number, 1);
    }

    #[test]
    fn parse_citation_markers_ignores_zero() {
        let markers = parse_citation_markers("Invalid [0] but valid [1].");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].number, 1);
    }

    #[test]
    fn parse_citation_markers_handles_double_digit() {
        let markers = parse_citation_markers("Citation [12] here.");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].number, 12);
    }

    #[test]
    fn parse_citation_markers_empty_text() {
        let markers = parse_citation_markers("");
        assert!(markers.is_empty());
    }

    #[test]
    fn parse_citation_markers_no_markers() {
        let markers = parse_citation_markers("No citations here.");
        assert!(markers.is_empty());
    }

    #[test]
    fn build_url_citations_maps_to_sources() {
        let sources = vec![
            WebSearchSource {
                url: "https://example.com/1".to_string(),
                title: "Source One".to_string(),
                snippet: None,
            },
            WebSearchSource {
                url: "https://example.com/2".to_string(),
                title: "Source Two".to_string(),
                snippet: Some("A snippet".to_string()),
            },
        ];

        let text = "According to [1] and [2], the answer is clear.";
        let annotations = build_url_citations(text, &sources);

        assert_eq!(annotations.len(), 2);

        match &annotations[0] {
            Annotation::UrlCitation {
                url,
                title,
                start_index,
                end_index,
            } => {
                assert_eq!(url, "https://example.com/1");
                assert_eq!(title, &Some("Source One".to_string()));
                assert_eq!(*start_index, 13);
                assert_eq!(*end_index, 16);
            }
            _ => panic!("expected UrlCitation"),
        }

        match &annotations[1] {
            Annotation::UrlCitation {
                url,
                title,
                start_index,
                end_index,
            } => {
                assert_eq!(url, "https://example.com/2");
                assert_eq!(title, &Some("Source Two".to_string()));
                assert_eq!(*start_index, 21);
                assert_eq!(*end_index, 24);
            }
            _ => panic!("expected UrlCitation"),
        }
    }

    #[test]
    fn build_url_citations_ignores_out_of_range() {
        let sources = vec![WebSearchSource {
            url: "https://example.com/1".to_string(),
            title: "Only Source".to_string(),
            snippet: None,
        }];

        let text = "Valid [1] but invalid [2] and [3].";
        let annotations = build_url_citations(text, &sources);

        // Only [1] should be matched
        assert_eq!(annotations.len(), 1);
        match &annotations[0] {
            Annotation::UrlCitation { url, .. } => {
                assert_eq!(url, "https://example.com/1");
            }
            _ => panic!("expected UrlCitation"),
        }
    }

    #[test]
    fn build_url_citations_empty_sources() {
        let sources: Vec<WebSearchSource> = vec![];
        let text = "Citation [1] with no sources.";
        let annotations = build_url_citations(text, &sources);
        assert!(annotations.is_empty());
    }
}
