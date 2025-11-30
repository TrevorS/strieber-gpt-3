//! Request types for the Responses API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ReasoningConfig, TextConfig, Tool, ToolChoice, Truncation};

/// Request body for POST /v1/responses
#[derive(Debug, Clone, Deserialize)]
pub struct CreateResponseRequest {
    /// Model ID (e.g., "gpt-4o", passed to the Chat Completions backend)
    pub model: String,

    /// Input to the model - either a string or array of input items
    #[serde(default)]
    pub input: Input,

    /// System/developer instructions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Tools the model may call
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// How the model selects tools
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Allow parallel tool calls
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Chain to a previous response for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls to process
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    /// Sampling temperature (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Nucleus sampling parameter
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Enable streaming response
    #[serde(default)]
    pub stream: bool,

    /// Store the response for later retrieval
    #[serde(default = "default_true")]
    pub store: bool,

    /// Reasoning configuration (for o-series models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Text format configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// Truncation strategy
    #[serde(default)]
    pub truncation: Truncation,

    /// Optional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

fn default_true() -> bool {
    true
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

// ============================================================================
// Input Types
// ============================================================================

/// Input can be a simple string or an array of input items.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    #[default]
    Empty,
    Text(String),
    Items(Vec<InputItem>),
}

/// An input item in the conversation.
/// These mirror the output items and can include reasoning from previous turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// A message from user, assistant, or system
    Message(MessageInput),

    /// Reasoning from a previous turn (for multi-turn with reasoning models)
    Reasoning(ReasoningInput),

    /// Output from a function call (tool result)
    FunctionCallOutput(FunctionCallOutputInput),

    /// Output from a custom tool call
    CustomToolCallOutput(CustomToolCallOutputInput),

    /// A function call from the assistant (to include in context)
    FunctionCall(FunctionCallInput),

    /// Computer call output (screenshot result, etc.)
    ComputerCallOutput(ComputerCallOutputInput),
}

// ============================================================================
// Message Input
// ============================================================================

/// A message input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageInput {
    pub role: Role,
    pub content: MessageContent,
}

/// Message content - string or array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content
    InputText { text: String },
    /// Image content (base64 or URL)
    InputImage { image_url: ImageUrl },
    /// File content
    InputFile { file: FileInput },
}

/// Image URL or base64 data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// File input reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
}

/// Role in the conversation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Developer,
}

// ============================================================================
// Reasoning Input (for multi-turn with reasoning models)
// ============================================================================

/// Reasoning input from a previous turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningInput {
    /// ID of the reasoning item
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Reasoning content
    #[serde(default)]
    pub content: Vec<ReasoningContentInput>,
    /// Encrypted content for stateless multi-turn
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
}

/// Content within reasoning input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningContentInput {
    ReasoningText { text: String },
    Redacted {},
}

// ============================================================================
// Function Call Input/Output
// ============================================================================

/// Function call from assistant (to include in context for multi-turn).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallInput {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Function call output (tool result) to send back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallOutputInput {
    /// The call_id from the function_call output item
    pub call_id: String,
    /// The result of the function call (typically JSON string)
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Custom tool call output to send back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCallOutputInput {
    /// The call_id from the custom_tool_call output item
    pub call_id: String,
    /// The result of the tool call
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

// ============================================================================
// Computer Call Output Input
// ============================================================================

/// Computer call output to send back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerCallOutputInput {
    /// The call_id from the computer_call output item
    pub call_id: String,
    /// The output (e.g., screenshot image)
    pub output: ComputerOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Acknowledged safety checks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acknowledged_safety_checks: Vec<String>,
}

/// Computer call output data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerOutput {
    Screenshot { image_url: String },
    Error { error: String },
}
