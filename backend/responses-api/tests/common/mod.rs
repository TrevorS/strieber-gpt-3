//! Common utilities for integration tests.
//!
//! These tests run against the actual llama.cpp server in the docker network.
//! They are conditionally compiled and only run when LLAMA_INTEGRATION_URL is set.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

/// Get the llama.cpp server URL from environment, defaulting to docker network URL.
pub fn llama_url() -> Option<String> {
    env::var("LLAMA_INTEGRATION_URL")
        .or_else(|_| env::var("LLAMA_URL"))
        .ok()
}

/// Get the responses-api URL from environment.
pub fn responses_api_url() -> Option<String> {
    env::var("RESPONSES_API_URL").ok()
}

/// Check if integration tests should run.
pub fn should_run_integration_tests() -> bool {
    llama_url().is_some() || responses_api_url().is_some()
}

/// Create an HTTP client configured for integration tests.
pub fn create_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(120)) // Long timeout for LLM responses
        .build()
        .expect("failed to create HTTP client")
}

/// Simple chat completion request for testing.
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

/// Chat completion response for testing.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
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

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Skip test if integration tests are not enabled.
#[macro_export]
macro_rules! skip_if_no_integration {
    () => {
        if !crate::common::should_run_integration_tests() {
            eprintln!("Skipping integration test: LLAMA_INTEGRATION_URL or RESPONSES_API_URL not set");
            return;
        }
    };
}
