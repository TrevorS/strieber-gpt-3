//! Shared utilities for integration tests.
//!
//! Run with: MODELS_CONFIG='{"models":[{"id":"test","url":"http://localhost:8000"}]}' cargo test

#![allow(dead_code)]

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

/// Model configuration from MODELS_CONFIG JSON.
#[derive(Debug, Deserialize)]
struct ModelsConfig {
    models: Vec<ModelConfig>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[allow(dead_code)]
    id: String,
    url: String,
}

/// Get the Chat Completions URL from MODELS_CONFIG (first model's URL).
pub fn chat_completions_url() -> Option<String> {
    let json = env::var("MODELS_CONFIG").ok()?;
    let config: ModelsConfig = serde_json::from_str(&json).ok()?;
    config.models.first().map(|m| m.url.clone())
}

pub fn responses_api_url() -> Option<String> {
    env::var("RESPONSES_API_URL").ok()
}

pub fn should_run_integration_tests() -> bool {
    chat_completions_url().is_some() || responses_api_url().is_some()
}

pub fn create_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to create HTTP client")
}

#[derive(Debug, Serialize)]
pub struct SimpleChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChoiceMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChoiceMessage {
    pub role: String,
    pub content: Option<String>,
}

/// Usage for Chat Completions API (prompt_tokens, completion_tokens)
#[derive(Debug, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Usage for Responses API (input_tokens, output_tokens)
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// Shared Responses API types
// ============================================================================

/// Input types for Responses API requests.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<InputItem>),
}

/// Input item for Responses API (requires type tag).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message(InputMessage),
}

#[derive(Debug, Serialize)]
pub struct InputMessage {
    pub role: String,
    pub content: String,
}

/// Create response request (unified version supporting both input types).
/// Use Input::Text() for simple string inputs or Input::Messages() for message arrays.
#[derive(Debug, Serialize)]
pub struct CreateResponseRequest {
    pub model: String,
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// Response object from Responses API.
#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub status: String,
    pub output: Vec<OutputItem>,
    // Optional fields that may not be present in all responses
    #[serde(default)]
    pub created_at: Option<i64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentPart>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    // Tool calling fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create HTTP client and get responses API URL for tests
pub fn setup_responses_test() -> Option<(Client, String)> {
    let url = responses_api_url()?;
    Some((create_client(), url))
}

/// Extract final assistant message text from response output
pub fn extract_final_message_text(output: &[OutputItem]) -> Option<&str> {
    output
        .iter()
        .rev()
        .find(|o| o.item_type == "message")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_deref())
}

/// Skip test if integration tests are not enabled.
#[macro_export]
macro_rules! skip_if_no_integration {
    () => {
        if !$crate::common::should_run_integration_tests() {
            eprintln!("Skipping: set MODELS_CONFIG or RESPONSES_API_URL");
            return;
        }
    };
}
