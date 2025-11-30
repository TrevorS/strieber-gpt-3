//! Integration tests for the Responses API service.
//!
//! Run these tests with:
//!   RESPONSES_API_URL=http://responses-api:8000 cargo test --test responses_api_integration
//!
//! Or from within the docker network:
//!   docker compose run --rm responses-api-test cargo test --test responses_api_integration

mod common;

use common::*;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
pub struct DeleteResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

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

/// Test health endpoint.
#[tokio::test]
async fn test_health_endpoint() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let resp = client
        .get(format!("{}/health", url))
        .send()
        .await
        .expect("failed to connect to responses-api");

    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

/// Test list models endpoint.
#[tokio::test]
async fn test_list_models() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let resp = client
        .get(format!("{}/v1/models", url))
        .send()
        .await
        .expect("failed to list models");

    assert!(resp.status().is_success());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(
        body["data"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    );
}

/// Test basic create response with text input.
#[tokio::test]
async fn test_create_response_text_input() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Say 'hello world' and nothing else.".to_string()),
        instructions: None,
        max_output_tokens: Some(100), // Reasoning models need more tokens for thinking
        temperature: Some(0.0),
        store: Some(false),
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    assert!(
        resp.status().is_success(),
        "create response failed: {}",
        resp.status()
    );

    let body: Response = resp.json().await.expect("failed to parse response");

    assert!(body.id.starts_with("resp_"));
    assert_eq!(body.object, "response");
    assert_eq!(body.status, "completed");
    assert!(!body.output.is_empty());

    // Find the message output item
    let message = body
        .output
        .iter()
        .find(|o| o.item_type == "message")
        .expect("no message in output");

    assert_eq!(message.role.as_deref(), Some("assistant"));
    let content = message.content.as_ref().expect("no content");
    let text = content
        .iter()
        .find(|c| c.content_type == "output_text")
        .and_then(|c| c.text.as_ref())
        .expect("no text content");

    println!("Response: {}", text);
    assert!(text.to_lowercase().contains("hello"));
}

/// Test create response with message input.
#[tokio::test]
async fn test_create_response_message_input() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Items(vec![InputItem::Message(InputMessage {
            role: "user".to_string(),
            content: "What is 2+2? Answer with just the number.".to_string(),
        })]),
        instructions: None,
        max_output_tokens: Some(100), // Reasoning models need more tokens for thinking
        temperature: Some(0.0),
        store: Some(false),
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    assert!(resp.status().is_success());

    let body: Response = resp.json().await.unwrap();
    assert_eq!(body.status, "completed");

    let message = body
        .output
        .iter()
        .find(|o| o.item_type == "message")
        .unwrap();
    let text = message
        .content
        .as_ref()
        .unwrap()
        .iter()
        .find(|c| c.content_type == "output_text")
        .and_then(|c| c.text.as_ref())
        .unwrap();

    println!("Response: {}", text);
    assert!(text.contains("4"));
}

/// Test create response with system instructions.
#[tokio::test]
async fn test_create_response_with_instructions() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What is 2+2?".to_string()),
        instructions: Some("Always respond in Spanish.".to_string()),
        max_output_tokens: Some(50),
        temperature: Some(0.0),
        store: Some(false),
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    assert!(resp.status().is_success());

    let body: Response = resp.json().await.unwrap();
    let message = body
        .output
        .iter()
        .find(|o| o.item_type == "message")
        .unwrap();
    let text = message
        .content
        .as_ref()
        .unwrap()
        .iter()
        .find(|c| c.content_type == "output_text")
        .and_then(|c| c.text.as_ref())
        .unwrap();

    println!("Response (should be Spanish): {}", text);
    // Response should contain Spanish text or numbers
    assert!(text.contains("4") || text.contains("cuatro"));
}

/// Test store and retrieve response.
#[tokio::test]
async fn test_store_and_retrieve_response() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Create and store a response
    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Say 'stored'".to_string()),
        instructions: None,
        max_output_tokens: Some(10),
        temperature: Some(0.0),
        store: Some(true), // Store the response
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    assert!(resp.status().is_success());

    let created: Response = resp.json().await.unwrap();
    let response_id = created.id.clone();
    println!("Created response: {}", response_id);

    // Retrieve the stored response
    let get_resp = client
        .get(format!("{}/v1/responses/{}", url, response_id))
        .send()
        .await
        .expect("failed to get response");

    assert!(get_resp.status().is_success());

    let retrieved: Response = get_resp.json().await.unwrap();
    assert_eq!(retrieved.id, response_id);
    assert_eq!(retrieved.status, "completed");
}

/// Test delete response.
#[tokio::test]
async fn test_delete_response() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Create and store a response
    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Say 'to be deleted'".to_string()),
        instructions: None,
        max_output_tokens: Some(10),
        temperature: Some(0.0),
        store: Some(true),
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    let created: Response = resp.json().await.unwrap();
    let response_id = created.id.clone();

    // Delete the response
    let delete_resp = client
        .delete(format!("{}/v1/responses/{}", url, response_id))
        .send()
        .await
        .expect("failed to delete response");

    assert!(delete_resp.status().is_success());

    let deleted: DeleteResponse = delete_resp.json().await.unwrap();
    assert_eq!(deleted.id, response_id);
    assert!(deleted.deleted);

    // Try to get the deleted response - should 404
    let get_resp = client
        .get(format!("{}/v1/responses/{}", url, response_id))
        .send()
        .await
        .expect("failed to get response");

    assert_eq!(get_resp.status(), 404);
}

/// Test get non-existent response returns 404.
#[tokio::test]
async fn test_get_nonexistent_response() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let resp = client
        .get(format!("{}/v1/responses/resp_nonexistent_12345", url))
        .send()
        .await
        .expect("failed to send request");

    assert_eq!(resp.status(), 404);

    let body: ErrorResponse = resp.json().await.unwrap();
    assert_eq!(body.error.error_type, "not_found");
}

/// Test streaming response returns SSE events.
#[tokio::test]
async fn test_streaming_response() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Say hello",
        "stream": true,
        "max_output_tokens": 20
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    // Streaming should return 200 with text/event-stream content type
    assert!(
        resp.status().is_success(),
        "expected success, got {}",
        resp.status()
    );

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "expected text/event-stream, got {}",
        content_type
    );

    // Read the SSE stream
    let body = resp.text().await.unwrap();
    println!("SSE Response:\n{}", body);

    // Should contain response.created event
    assert!(
        body.contains("response.created") || body.contains("response.completed"),
        "expected SSE events in response"
    );
}

/// Test usage tokens are reported correctly.
#[tokio::test]
async fn test_usage_reporting() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Hi".to_string()),
        instructions: None,
        max_output_tokens: Some(5),
        temperature: Some(0.0),
        store: Some(false),
        tools: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create response");

    assert!(resp.status().is_success());

    let body: Response = resp.json().await.unwrap();

    let usage = body.usage.as_ref().expect("usage should be present");
    assert!(usage.input_tokens > 0, "input_tokens should be > 0");
    assert!(usage.output_tokens > 0, "output_tokens should be > 0");
    assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);

    println!(
        "Usage: {} input + {} output = {} total",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    );
}
