//! Integration tests for llama.cpp server.
//!
//! Run these tests with:
//!   LLAMA_INTEGRATION_URL=http://llama-server:8000 cargo test --test llama_integration
//!
//! Or from within the docker network:
//!   docker compose run --rm responses-api cargo test --test llama_integration

mod common;

use common::{ChatMessage, SimpleChatRequest, create_client, llama_url};

/// Test basic connectivity to llama.cpp server.
#[tokio::test]
async fn test_llama_health() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    let resp = client
        .get(format!("{}/health", url))
        .send()
        .await
        .expect("failed to connect to llama server");

    assert!(
        resp.status().is_success(),
        "llama server health check failed: {}",
        resp.status()
    );
}

/// Test listing models from llama.cpp.
#[tokio::test]
async fn test_list_models() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    let resp = client
        .get(format!("{}/v1/models", url))
        .send()
        .await
        .expect("failed to list models");

    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.expect("failed to parse response");
    assert_eq!(body["object"], "list");
    assert!(
        body["data"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    );

    println!(
        "Available models: {}",
        serde_json::to_string_pretty(&body["data"]).unwrap()
    );
}

/// Test basic chat completion.
#[tokio::test]
async fn test_basic_chat_completion() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    let req = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Say 'hello' and nothing else.".to_string(),
        }],
        max_tokens: Some(100), // Reasoning models need more tokens for thinking
        temperature: Some(0.0),
    };

    let resp = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send chat request");

    assert!(
        resp.status().is_success(),
        "chat completion failed: {}",
        resp.status()
    );

    let body: common::ChatCompletionResponse = resp.json().await.expect("failed to parse response");

    assert!(!body.choices.is_empty());
    let content = body.choices[0]
        .message
        .content
        .as_ref()
        .expect("no content");
    println!("Response: {}", content);
    assert!(content.to_lowercase().contains("hello"));
}

/// Test chat completion with system message.
#[tokio::test]
async fn test_chat_with_system_message() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    let req = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant. Always respond in exactly 3 words."
                    .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
        ],
        max_tokens: Some(20),
        temperature: Some(0.0),
    };

    let resp = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send chat request");

    assert!(resp.status().is_success());

    let body: common::ChatCompletionResponse = resp.json().await.expect("failed to parse response");
    let content = body.choices[0]
        .message
        .content
        .as_ref()
        .expect("no content");
    println!("Response: {}", content);
}

/// Test multi-turn conversation.
#[tokio::test]
async fn test_multi_turn_conversation() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    // First turn
    let req1 = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "My name is Alice. Remember this.".to_string(),
        }],
        max_tokens: Some(150), // Reasoning models need more tokens for thinking
        temperature: Some(0.0),
    };

    let resp1 = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req1)
        .send()
        .await
        .expect("failed to send first turn");

    assert!(resp1.status().is_success());
    let body1: common::ChatCompletionResponse = resp1.json().await.unwrap();
    let assistant_reply = body1.choices[0].message.content.clone().unwrap_or_default();

    // Second turn with context
    let req2 = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: "My name is Alice. Remember this.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: assistant_reply,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is my name?".to_string(),
            },
        ],
        max_tokens: Some(150), // Reasoning models need more tokens for thinking
        temperature: Some(0.0),
    };

    let resp2 = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req2)
        .send()
        .await
        .expect("failed to send second turn");

    assert!(resp2.status().is_success());
    let body2: common::ChatCompletionResponse = resp2.json().await.unwrap();
    let content = body2.choices[0]
        .message
        .content
        .as_ref()
        .expect("no content");
    println!("Response: {}", content);
    assert!(content.to_lowercase().contains("alice"));
}

/// Test that the server handles invalid requests gracefully.
#[tokio::test]
async fn test_invalid_request() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    // Empty messages should fail
    let req = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![],
        max_tokens: None,
        temperature: None,
    };

    let resp = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    // Should return an error status
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error for empty messages, got: {}",
        resp.status()
    );
}

/// Test token usage reporting.
#[tokio::test]
async fn test_usage_reporting() {
    skip_if_no_integration!();

    let client = create_client();
    let url = llama_url().unwrap();

    let req = SimpleChatRequest {
        model: "gpt-oss-120b".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }],
        max_tokens: Some(5),
        temperature: Some(0.0),
    };

    let resp = client
        .post(format!("{}/v1/chat/completions", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    assert!(resp.status().is_success());

    let body: common::ChatCompletionResponse = resp.json().await.unwrap();

    // Verify usage is reported
    assert!(body.usage.prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(
        body.usage.completion_tokens > 0,
        "completion_tokens should be > 0"
    );
    assert_eq!(
        body.usage.total_tokens,
        body.usage.prompt_tokens + body.usage.completion_tokens
    );

    println!(
        "Usage: {} prompt + {} completion = {} total",
        body.usage.prompt_tokens, body.usage.completion_tokens, body.usage.total_tokens
    );
}
