//! Shared utilities for integration tests.
//!
//! Run with: LLAMA_INTEGRATION_URL=http://localhost:8000 cargo test

#![allow(dead_code)]

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

pub fn llama_url() -> Option<String> {
    env::var("LLAMA_INTEGRATION_URL")
        .or_else(|_| env::var("LLAMA_URL"))
        .ok()
}

pub fn responses_api_url() -> Option<String> {
    env::var("RESPONSES_API_URL").ok()
}

pub fn should_run_integration_tests() -> bool {
    llama_url().is_some() || responses_api_url().is_some()
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
        if !$crate::common::should_run_integration_tests() {
            eprintln!("Skipping: set LLAMA_INTEGRATION_URL or RESPONSES_API_URL");
            return;
        }
    };
}
