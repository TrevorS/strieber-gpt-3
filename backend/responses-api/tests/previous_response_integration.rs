//! Integration tests for previous_response_id (multi-turn conversations).
//!
//! Run these tests with:
//!   RESPONSES_API_URL=http://responses-api:8000 cargo test --test previous_response_integration
//!
//! Or from within the docker network:
//!   docker compose run --rm responses-api-test cargo test --test previous_response_integration
//!
//! These tests cover:
//! - Non-streaming context chaining
//! - Streaming response storage and context chaining (regression test for bug where
//!   streaming responses with store:true were not being persisted)

mod common;

use common::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Test two-turn conversation using previous_response_id.
#[tokio::test]
async fn test_two_turn_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Turn 1: Ask a question
    let req1 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("My name is Alice. Please remember that.".to_string()),
        instructions: Some("You are a helpful assistant.".to_string()),
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true), // Must store to use previous_response_id
        stream: None,
        tools: None,
        previous_response_id: None,
    };

    let resp1 = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .send()
        .await
        .expect("failed to create first response");

    assert!(
        resp1.status().is_success(),
        "first response failed: {}",
        resp1.status()
    );

    let body1: Response = resp1.json().await.expect("failed to parse first response");
    assert_eq!(body1.status, "completed");
    println!("First response ID: {}", body1.id);

    // Turn 2: Follow up, referencing the previous response
    let req2 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What is my name?".to_string()),
        instructions: None, // Should inherit from first request
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: None,
        tools: None,
        previous_response_id: Some(body1.id.clone()),
    };

    let resp2 = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .send()
        .await
        .expect("failed to create second response");

    assert!(
        resp2.status().is_success(),
        "second response failed: {}",
        resp2.status()
    );

    let body2: Response = resp2.json().await.expect("failed to parse second response");
    assert_eq!(body2.status, "completed");
    assert_eq!(body2.previous_response_id, Some(body1.id.clone()));

    // The model should remember Alice's name
    let text = extract_final_message_text(&body2.output).expect("no message in output");
    println!("Second response: {}", text);
    assert!(
        text.to_lowercase().contains("alice"),
        "Model should remember name 'Alice', got: {}",
        text
    );
}

/// Test previous_response_id with non-existent ID returns 400.
#[tokio::test]
async fn test_previous_response_id_not_found() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Hello".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: None,
        previous_response_id: Some("resp_nonexistent_12345".to_string()),
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    assert_eq!(
        resp.status().as_u16(),
        400,
        "expected 400 Bad Request, got {}",
        resp.status()
    );

    let error: ErrorResponse = resp.json().await.expect("failed to parse error response");
    assert_eq!(error.error.error_type, "invalid_request_error");
    assert!(
        error.error.message.contains("not found"),
        "error message should mention 'not found': {}",
        error.error.message
    );
}

/// Test three-turn conversation chain.
#[tokio::test]
async fn test_three_turn_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Turn 1: Set up context
    let req1 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("I have a dog named Spot.".to_string()),
        instructions: Some("You are a helpful assistant with perfect memory.".to_string()),
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: None,
        tools: None,
        previous_response_id: None,
    };

    let resp1: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .send()
        .await
        .expect("turn 1 failed")
        .json()
        .await
        .expect("failed to parse turn 1");

    // Turn 2: Add more context
    let req2 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("I also have a cat named Whiskers.".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: None,
        tools: None,
        previous_response_id: Some(resp1.id.clone()),
    };

    let resp2: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .send()
        .await
        .expect("turn 2 failed")
        .json()
        .await
        .expect("failed to parse turn 2");

    // Turn 3: Query both pieces of information
    let req3 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What are the names of my pets?".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: None,
        tools: None,
        previous_response_id: Some(resp2.id.clone()),
    };

    let resp3: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req3)
        .send()
        .await
        .expect("turn 3 failed")
        .json()
        .await
        .expect("failed to parse turn 3");

    // Verify chain linkage
    assert_eq!(resp3.previous_response_id, Some(resp2.id.clone()));

    // Model should remember both pets
    let text = extract_final_message_text(&resp3.output).expect("no message");
    let lower = text.to_lowercase();
    println!("Third response: {}", text);
    assert!(
        lower.contains("spot"),
        "Should remember dog 'Spot': {}",
        text
    );
    assert!(
        lower.contains("whiskers"),
        "Should remember cat 'Whiskers': {}",
        text
    );
}

/// Test instruction inheritance from chain.
#[tokio::test]
async fn test_instruction_inheritance() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Turn 1: Set specific instructions
    let req1 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Hello".to_string()),
        instructions: Some("Always respond in ALL CAPS.".to_string()),
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: None,
        tools: None,
        previous_response_id: None,
    };

    let resp1: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .send()
        .await
        .expect("turn 1 failed")
        .json()
        .await
        .expect("failed to parse turn 1");

    // Turn 2: No instructions specified - should inherit from turn 1
    let req2 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What is 2 + 2?".to_string()),
        instructions: None, // Should inherit ALL CAPS instruction
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: None,
        previous_response_id: Some(resp1.id.clone()),
    };

    let resp2: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .send()
        .await
        .expect("turn 2 failed")
        .json()
        .await
        .expect("failed to parse turn 2");

    let text = extract_final_message_text(&resp2.output).expect("no message");
    println!("Response (should be caps): {}", text);

    // Response should be in all caps (inherited instruction)
    // Check that most alphabetic characters are uppercase
    let alpha_chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
    let upper_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
    let upper_ratio = upper_count as f64 / alpha_chars.len().max(1) as f64;

    assert!(
        upper_ratio > 0.7,
        "Response should be mostly uppercase (inherited instruction), got: {} ({}% upper)",
        text,
        (upper_ratio * 100.0) as u32
    );
}

// ============================================================================
// Streaming Response Storage Tests
// ============================================================================
// These tests verify that streaming responses with store:true are properly
// persisted and can be used with previous_response_id.
//
// This is a regression test for a bug where streaming responses were not
// being stored, causing context chaining to fail silently.

/// Test that streaming responses with store:true are persisted.
/// This is a regression test for the bug where streaming responses weren't stored.
#[tokio::test]
async fn test_streaming_response_is_stored() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Turn 1: Send a streaming request with store:true
    let req1 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("My favorite programming language is Rust. Remember that.".to_string()),
        instructions: Some("You are a helpful assistant.".to_string()),
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: Some(true), // Streaming!
        tools: None,
        previous_response_id: None,
    };

    let resp1 = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .send()
        .await
        .expect("failed to send streaming request");

    assert!(
        resp1.status().is_success(),
        "streaming request failed: {}",
        resp1.status()
    );

    // Read the SSE stream body and extract response ID
    let body = resp1.text().await.expect("failed to read SSE body");
    let resp1_id =
        extract_response_id_from_sse(&body).expect("could not extract response ID from SSE stream");

    println!("Streaming response ID: {}", resp1_id);

    // Turn 2: Follow up using the streaming response's ID
    let req2 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What is my favorite programming language?".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: Some(false), // Non-streaming for easier assertion
        tools: None,
        previous_response_id: Some(resp1_id.clone()),
    };

    let resp2 = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .send()
        .await
        .expect("failed to send follow-up request");

    assert!(
        resp2.status().is_success(),
        "follow-up request failed (streaming response may not have been stored): {}",
        resp2.status()
    );

    let body2: Response = resp2
        .json()
        .await
        .expect("failed to parse follow-up response");
    assert_eq!(body2.status, "completed");
    assert_eq!(body2.previous_response_id, Some(resp1_id));

    // The model should remember the favorite language
    let text = extract_final_message_text(&body2.output).expect("no message in output");
    println!("Follow-up response: {}", text);
    assert!(
        text.to_lowercase().contains("rust"),
        "Model should remember 'Rust' from streaming response, got: {}",
        text
    );
}

/// Test multi-turn conversation mixing streaming and non-streaming.
#[tokio::test]
async fn test_mixed_streaming_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Turn 1: Non-streaming
    let req1 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("I live in Portland.".to_string()),
        instructions: Some("You are a helpful assistant with perfect memory.".to_string()),
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: Some(false),
        tools: None,
        previous_response_id: None,
    };

    let resp1: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .send()
        .await
        .expect("turn 1 failed")
        .json()
        .await
        .expect("failed to parse turn 1");

    println!("Turn 1 (non-streaming) ID: {}", resp1.id);

    // Turn 2: Streaming, referencing turn 1
    let req2 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("I work as a software engineer.".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(true),
        stream: Some(true), // Streaming!
        tools: None,
        previous_response_id: Some(resp1.id.clone()),
    };

    let resp2_raw = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .send()
        .await
        .expect("turn 2 failed");

    let body2 = resp2_raw.text().await.expect("failed to read turn 2 body");
    let resp2_id =
        extract_response_id_from_sse(&body2).expect("could not extract ID from turn 2 SSE");

    println!("Turn 2 (streaming) ID: {}", resp2_id);

    // Turn 3: Non-streaming, should have context from both previous turns
    let req3 = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Where do I live and what do I do for work?".to_string()),
        instructions: None,
        max_output_tokens: Some(100),
        temperature: Some(0.0),
        store: Some(false),
        stream: Some(false),
        tools: None,
        previous_response_id: Some(resp2_id.clone()),
    };

    let resp3: Response = client
        .post(format!("{}/v1/responses", url))
        .json(&req3)
        .send()
        .await
        .expect("turn 3 failed")
        .json()
        .await
        .expect("failed to parse turn 3");

    let text = extract_final_message_text(&resp3.output).expect("no message");
    let lower = text.to_lowercase();
    println!("Turn 3 response: {}", text);

    assert!(
        lower.contains("portland"),
        "Should remember 'Portland' from turn 1: {}",
        text
    );
    assert!(
        lower.contains("software") || lower.contains("engineer"),
        "Should remember 'software engineer' from turn 2: {}",
        text
    );
}
