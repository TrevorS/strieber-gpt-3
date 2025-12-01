//! Integration tests for previous_response_id (multi-turn conversations).
//!
//! Run these tests with:
//!   RESPONSES_API_URL=http://responses-api:8000 cargo test --test previous_response_integration
//!
//! Or from within the docker network:
//!   docker compose run --rm responses-api-test cargo test --test previous_response_integration

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
