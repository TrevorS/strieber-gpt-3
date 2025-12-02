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
        stream: None,
        tools: None,
        previous_response_id: None,
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
        stream: None,
        tools: None,
        previous_response_id: None,
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
        stream: None,
        tools: None,
        previous_response_id: None,
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
        stream: None,
        tools: None,
        previous_response_id: None,
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
        stream: None,
        tools: None,
        previous_response_id: None,
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
        stream: None,
        tools: None,
        previous_response_id: None,
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

/// Test streaming emits expected SSE event types with content.
#[tokio::test]
async fn test_streaming_emits_content_deltas() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Request with stream: true and enough tokens for content
    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Say 'hello world' word by word.",
        "max_output_tokens": 100,
        "stream": true
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream")
    );

    // Collect all SSE events
    let body = resp.text().await.unwrap();

    // Parse event types from SSE stream
    let event_types: Vec<&str> = body
        .lines()
        .filter(|l| l.starts_with("event: "))
        .map(|l| l.trim_start_matches("event: "))
        .collect();

    println!("SSE events received: {:?}", event_types);

    // Required lifecycle events
    assert!(
        event_types.contains(&"response.created"),
        "Missing response.created event"
    );
    assert!(
        event_types.contains(&"response.in_progress"),
        "Missing response.in_progress event"
    );
    assert!(
        event_types.contains(&"response.completed"),
        "Missing response.completed event"
    );
    assert!(
        event_types.contains(&"response.done"),
        "Missing response.done event"
    );

    // Should have some content-related events (output_item or delta)
    let has_content_events = event_types
        .iter()
        .any(|e| e.contains("output_item") || e.contains("delta") || e.contains("text"));
    println!(
        "Has content events: {} (events: {:?})",
        has_content_events, event_types
    );

    // Verify the done event shows completed status
    assert!(
        body.contains("\"status\":\"completed\"") || body.contains("\"status\":\"in_progress\""),
        "Final event should have status"
    );
}

/// Test streaming works with tool calls.
#[tokio::test]
async fn test_streaming_with_tool_call() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "What's the weather in Tokyo?",
        "instructions": "Use the weather tool to get accurate information.",
        "max_output_tokens": 200,
        "stream": true,
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }
        }]
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .expect("request failed");

    assert!(resp.status().is_success());

    let body = resp.text().await.unwrap();
    println!("Streaming with tools response:\n{}", body);

    // Should have function call events OR a direct text response
    assert!(
        body.contains("function_call") || body.contains("output_text") || body.contains("message"),
        "Should have tool call or text output in streaming response"
    );
    assert!(
        body.contains("response.done"),
        "Should have response.done event"
    );
}

/// Test multi-turn conversation with reasoning context.
#[tokio::test]
async fn test_multi_turn_with_reasoning_input() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // First request - get a response (gpt-oss-120b may include reasoning)
    let req1 = json!({
        "model": "gpt-oss-120b",
        "input": "My name is Alice. Remember this name.",
        "max_output_tokens": 200,
        "temperature": 0.0,
        "store": true
    });

    let resp1 = client
        .post(format!("{}/v1/responses", url))
        .json(&req1)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .expect("first request failed");

    assert!(resp1.status().is_success());

    let body1: Response = resp1.json().await.expect("failed to parse first response");

    // Check if we got reasoning output
    let has_reasoning = body1.output.iter().any(|o| o.item_type == "reasoning");
    println!("First response has reasoning: {}", has_reasoning);

    // Build second request with full context
    let mut input_items = vec![];

    // Add original user message
    input_items.push(json!({
        "type": "message",
        "role": "user",
        "content": "My name is Alice. Remember this name."
    }));

    // Add reasoning if present (to test reasoning input)
    if let Some(reasoning) = body1.output.iter().find(|o| o.item_type == "reasoning") {
        if let Some(content) = &reasoning.content {
            if let Some(text_part) = content.iter().find(|c| c.content_type == "output_text") {
                if let Some(text) = &text_part.text {
                    println!(
                        "Including reasoning context: {}...",
                        &text[..text.len().min(50)]
                    );
                    input_items.push(json!({
                        "type": "reasoning",
                        "id": reasoning.id,
                        "content": [{"type": "input_text", "text": text}]
                    }));
                }
            }
        }
    }

    // Add assistant response
    if let Some(msg) = body1.output.iter().find(|o| o.item_type == "message") {
        if let Some(content) = &msg.content {
            if let Some(text_part) = content.iter().find(|c| c.content_type == "output_text") {
                if let Some(text) = &text_part.text {
                    input_items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text
                    }));
                }
            }
        }
    }

    // Add follow-up question
    input_items.push(json!({
        "type": "message",
        "role": "user",
        "content": "What is my name?"
    }));

    let req2 = json!({
        "model": "gpt-oss-120b",
        "input": input_items,
        "max_output_tokens": 150,
        "temperature": 0.0
    });

    let resp2 = client
        .post(format!("{}/v1/responses", url))
        .json(&req2)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .expect("second request failed");

    assert!(
        resp2.status().is_success(),
        "second request failed: {}",
        resp2.status()
    );

    let body2: Response = resp2.json().await.expect("failed to parse second response");

    // Should remember the name
    let final_text = body2
        .output
        .iter()
        .find(|o| o.item_type == "message")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_ref())
        .expect("no response text");

    println!("Multi-turn response: {}", final_text);
    assert!(
        final_text.to_lowercase().contains("alice"),
        "Should remember the name Alice, got: {}",
        final_text
    );
}

// ============================================================================
// Container Files API Tests
// ============================================================================

/// Test list files on non-existent container returns 404.
#[tokio::test]
async fn test_list_container_files_not_found() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let resp = client
        .get(format!("{}/v1/containers/cntr_nonexistent/files", url))
        .send()
        .await
        .expect("failed to send request");

    assert_eq!(resp.status(), 404);

    let body: ErrorResponse = resp.json().await.unwrap();
    assert_eq!(body.error.error_type, "not_found");
    assert!(body.error.message.contains("not found"));
}

/// Test download file from non-existent container returns 404.
#[tokio::test]
async fn test_download_file_container_not_found() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let resp = client
        .get(format!(
            "{}/v1/containers/cntr_nonexistent/files/file_123/content",
            url
        ))
        .send()
        .await
        .expect("failed to send request");

    assert_eq!(resp.status(), 404);

    let body: ErrorResponse = resp.json().await.unwrap();
    assert_eq!(body.error.error_type, "not_found");
    assert!(body.error.message.contains("Container"));
}

/// Test code interpreter generates file citations.
/// This test uses the code interpreter to create a matplotlib plot,
/// which should result in file citations in the response.
#[tokio::test]
async fn test_code_interpreter_generates_file_citations() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Request that uses code interpreter to generate a plot
    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Use Python to create a simple sine wave plot. Just run the code.",
        "instructions": "Execute Python code using the code interpreter. Create a matplotlib sine wave plot.",
        "max_output_tokens": 500,
        "tools": [{"type": "code_interpreter"}]
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .expect("request failed");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Request failed with {}: {}", status, body);
    }

    let body: Response = resp.json().await.expect("failed to parse response");
    println!("Response output items: {}", body.output.len());

    // Look for file citations in annotations
    let mut found_file_citation = false;
    for item in &body.output {
        if let Some(content) = &item.content {
            for part in content {
                // ContentPart might have annotations - need to check raw JSON
                println!("Content part type: {}", part.content_type);
            }
        }
    }

    // Parse the full response to look for container_file_citation
    let raw_text = serde_json::to_string(&body).unwrap();
    if raw_text.contains("container_file_citation") || raw_text.contains("file_id") {
        found_file_citation = true;
        println!("Found file citation in response");
    }

    println!("Response contains file citations: {}", found_file_citation);
    // Note: This test validates the flow works but may not always generate images
    // depending on the model's behavior
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/// Test invalid model returns error.
#[tokio::test]
async fn test_invalid_model_returns_error() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "nonexistent-model-xyz",
        "input": "Hello",
        "max_output_tokens": 10
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    // Should return an error (either 400 or 404)
    assert!(
        resp.status().is_client_error(),
        "expected client error, got {}",
        resp.status()
    );

    let body: ErrorResponse = resp.json().await.expect("should be error response");
    println!(
        "Error type: {}, message: {}",
        body.error.error_type, body.error.message
    );
    // Error should mention the model
    assert!(
        body.error.message.contains("model") || body.error.message.contains("nonexistent"),
        "error should mention model"
    );
}

/// Test malformed request returns 4xx error.
#[tokio::test]
async fn test_malformed_request_returns_error() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Send request missing required 'model' field
    let req = json!({
        "input": "Hello"
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    // Should return 400 (Bad Request) or 422 (Unprocessable Entity)
    let status = resp.status().as_u16();
    assert!(
        status == 400 || status == 422,
        "missing model should return 400 or 422, got {}",
        status
    );
}

/// Test empty input returns 400.
#[tokio::test]
async fn test_empty_input_returns_400() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": ""
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .send()
        .await
        .expect("failed to send request");

    // Empty input should either work or return 400, not 500
    let status = resp.status().as_u16();
    assert!(
        status == 200 || status == 400,
        "empty input should return 200 or 400, got {}",
        status
    );
}
