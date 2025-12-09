//! Integration tests for the Conversations API.
//!
//! Run these tests with:
//!   RESPONSES_API_URL=http://responses-api:8000 cargo test --test conversation_api_integration
//!
//! These tests cover:
//! - Conversation CRUD operations
//! - Conversation items management
//! - Streaming responses with conversation context
//! - Non-streaming responses with conversation context

mod common;

use common::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Conversation API Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct CreateConversationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<ConversationInputItem>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationInputItem {
    Message(ConversationMessage),
}

#[derive(Debug, Serialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct ConversationItem {
    pub id: String,
    pub status: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ListResponse<T> {
    pub object: String,
    pub data: Vec<T>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize)]
pub struct ConversationDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateItemsRequest {
    pub items: Vec<ConversationInputItem>,
}

#[derive(Debug, Serialize)]
pub struct ConversationResponseRequest {
    pub model: String,
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    pub conversation: ConversationParam,
}

#[derive(Debug, Serialize)]
pub struct ConversationParam {
    pub id: String,
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

// ============================================================================
// Conversation CRUD Tests
// ============================================================================

/// Test creating a conversation without items.
#[tokio::test]
async fn test_create_conversation_empty() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    let req = CreateConversationRequest {
        metadata: None,
        items: None,
    };

    let resp = client
        .post(format!("{}/v1/conversations", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create conversation");

    assert!(
        resp.status().is_success(),
        "create conversation failed: {}",
        resp.status()
    );

    let conv: Conversation = resp.json().await.expect("failed to parse conversation");
    assert!(conv.id.starts_with("conv_"));
    assert_eq!(conv.object, "conversation");
}

/// Test creating a conversation with initial items.
#[tokio::test]
async fn test_create_conversation_with_items() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    let req = CreateConversationRequest {
        metadata: None,
        items: Some(vec![ConversationInputItem::Message(ConversationMessage {
            role: "user".to_string(),
            content: "Hello, I'm starting a conversation.".to_string(),
        })]),
    };

    let resp = client
        .post(format!("{}/v1/conversations", url))
        .json(&req)
        .send()
        .await
        .expect("failed to create conversation");

    assert!(resp.status().is_success());

    let conv: Conversation = resp.json().await.expect("failed to parse conversation");
    assert!(conv.id.starts_with("conv_"));

    // Verify item was added
    let items_resp = client
        .get(format!("{}/v1/conversations/{}/items", url, conv.id))
        .send()
        .await
        .expect("failed to list items");

    let items: ListResponse<ConversationItem> =
        items_resp.json().await.expect("failed to parse items");
    assert_eq!(items.data.len(), 1);
}

/// Test getting a conversation.
#[tokio::test]
async fn test_get_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation
    let create_req = CreateConversationRequest {
        metadata: None,
        items: None,
    };

    let create_resp: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&create_req)
        .send()
        .await
        .expect("failed to create")
        .json()
        .await
        .expect("failed to parse");

    // Get the conversation
    let get_resp = client
        .get(format!("{}/v1/conversations/{}", url, create_resp.id))
        .send()
        .await
        .expect("failed to get conversation");

    assert!(get_resp.status().is_success());

    let conv: Conversation = get_resp.json().await.expect("failed to parse");
    assert_eq!(conv.id, create_resp.id);
}

/// Test getting a non-existent conversation returns 404.
#[tokio::test]
async fn test_get_nonexistent_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    let resp = client
        .get(format!("{}/v1/conversations/conv_nonexistent_12345", url))
        .send()
        .await
        .expect("failed to send request");

    assert_eq!(resp.status().as_u16(), 404);
}

/// Test deleting a conversation.
#[tokio::test]
async fn test_delete_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation
    let create_req = CreateConversationRequest {
        metadata: None,
        items: None,
    };

    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&create_req)
        .send()
        .await
        .expect("failed to create")
        .json()
        .await
        .expect("failed to parse");

    // Delete it
    let del_resp = client
        .delete(format!("{}/v1/conversations/{}", url, conv.id))
        .send()
        .await
        .expect("failed to delete");

    assert!(del_resp.status().is_success());

    let deleted: ConversationDeleted = del_resp.json().await.expect("failed to parse");
    assert_eq!(deleted.id, conv.id);
    assert!(deleted.deleted);

    // Verify it's gone
    let get_resp = client
        .get(format!("{}/v1/conversations/{}", url, conv.id))
        .send()
        .await
        .expect("failed to get");

    assert_eq!(get_resp.status().as_u16(), 404);
}

// ============================================================================
// Conversation Items Tests
// ============================================================================

/// Test adding items to a conversation.
#[tokio::test]
async fn test_add_items_to_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create an empty conversation
    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&CreateConversationRequest {
            metadata: None,
            items: None,
        })
        .send()
        .await
        .expect("failed to create")
        .json()
        .await
        .expect("failed to parse");

    // Add items
    let add_req = CreateItemsRequest {
        items: vec![
            ConversationInputItem::Message(ConversationMessage {
                role: "user".to_string(),
                content: "First message".to_string(),
            }),
            ConversationInputItem::Message(ConversationMessage {
                role: "user".to_string(),
                content: "Second message".to_string(),
            }),
        ],
    };

    let add_resp = client
        .post(format!("{}/v1/conversations/{}/items", url, conv.id))
        .json(&add_req)
        .send()
        .await
        .expect("failed to add items");

    assert!(add_resp.status().is_success());

    let added: ListResponse<ConversationItem> = add_resp.json().await.expect("failed to parse");
    assert_eq!(added.data.len(), 2);

    // Verify items are in conversation
    let list_resp: ListResponse<ConversationItem> = client
        .get(format!("{}/v1/conversations/{}/items", url, conv.id))
        .send()
        .await
        .expect("failed to list")
        .json()
        .await
        .expect("failed to parse");

    assert_eq!(list_resp.data.len(), 2);
}

// ============================================================================
// Conversation + Response Integration Tests
// ============================================================================

/// Test non-streaming response with conversation context.
#[tokio::test]
async fn test_response_with_conversation_context() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation with initial context
    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&CreateConversationRequest {
            metadata: None,
            items: Some(vec![ConversationInputItem::Message(ConversationMessage {
                role: "user".to_string(),
                content: "My name is Charlie and I love hiking.".to_string(),
            })]),
        })
        .send()
        .await
        .expect("failed to create conversation")
        .json()
        .await
        .expect("failed to parse");

    println!("Created conversation: {}", conv.id);

    // Send a response request using the conversation
    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&ConversationResponseRequest {
            model: "gpt-oss-120b".to_string(),
            input: Input::Text("What is my name and what do I enjoy doing?".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            max_output_tokens: Some(100),
            temperature: Some(0.0),
            stream: Some(false),
            tools: None,
            conversation: ConversationParam {
                id: conv.id.clone(),
            },
        })
        .send()
        .await
        .expect("failed to send response request");

    assert!(
        resp.status().is_success(),
        "response request failed: {}",
        resp.status()
    );

    let response: Response = resp.json().await.expect("failed to parse response");
    let text = extract_final_message_text(&response.output).expect("no message in output");
    let lower = text.to_lowercase();

    println!("Response: {}", text);
    assert!(
        lower.contains("charlie"),
        "Should remember name 'Charlie': {}",
        text
    );
    assert!(
        lower.contains("hiking") || lower.contains("hike"),
        "Should remember 'hiking': {}",
        text
    );

    // Verify output was appended to conversation
    let items_resp: ListResponse<ConversationItem> = client
        .get(format!("{}/v1/conversations/{}/items", url, conv.id))
        .send()
        .await
        .expect("failed to list items")
        .json()
        .await
        .expect("failed to parse");

    // Should have: initial user message + new user input + assistant response
    assert!(
        items_resp.data.len() >= 2,
        "Expected at least 2 items, got {}",
        items_resp.data.len()
    );
}

/// Test streaming response with conversation context appends output.
#[tokio::test]
async fn test_streaming_response_with_conversation() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation with initial context
    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&CreateConversationRequest {
            metadata: None,
            items: Some(vec![ConversationInputItem::Message(ConversationMessage {
                role: "user".to_string(),
                content: "My favorite color is purple.".to_string(),
            })]),
        })
        .send()
        .await
        .expect("failed to create conversation")
        .json()
        .await
        .expect("failed to parse");

    println!("Created conversation: {}", conv.id);

    // Count items before streaming response
    let items_before: ListResponse<ConversationItem> = client
        .get(format!("{}/v1/conversations/{}/items", url, conv.id))
        .send()
        .await
        .expect("failed to list items")
        .json()
        .await
        .expect("failed to parse");

    let count_before = items_before.data.len();
    println!("Items before: {}", count_before);

    // Send a streaming response request using the conversation
    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&ConversationResponseRequest {
            model: "gpt-oss-120b".to_string(),
            input: Input::Text("What is my favorite color?".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            max_output_tokens: Some(100),
            temperature: Some(0.0),
            stream: Some(true), // Streaming!
            tools: None,
            conversation: ConversationParam {
                id: conv.id.clone(),
            },
        })
        .send()
        .await
        .expect("failed to send streaming request");

    assert!(
        resp.status().is_success(),
        "streaming request failed: {}",
        resp.status()
    );

    // Consume the SSE stream
    let body = resp.text().await.expect("failed to read SSE body");
    println!("SSE response received");

    // Verify the response mentions purple
    assert!(
        body.to_lowercase().contains("purple"),
        "Streaming response should mention 'purple'"
    );

    // Verify output was appended to conversation
    let items_after: ListResponse<ConversationItem> = client
        .get(format!("{}/v1/conversations/{}/items", url, conv.id))
        .send()
        .await
        .expect("failed to list items")
        .json()
        .await
        .expect("failed to parse");

    println!("Items after: {}", items_after.data.len());

    // Should have more items after streaming response
    assert!(
        items_after.data.len() > count_before,
        "Expected more items after streaming response. Before: {}, After: {}",
        count_before,
        items_after.data.len()
    );
}

/// Test multi-turn conversation using conversation API.
#[tokio::test]
async fn test_multi_turn_with_conversation_api() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation
    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&CreateConversationRequest {
            metadata: None,
            items: None,
        })
        .send()
        .await
        .expect("failed to create conversation")
        .json()
        .await
        .expect("failed to parse");

    println!("Created conversation: {}", conv.id);

    // Turn 1: Introduce context
    let resp1 = client
        .post(format!("{}/v1/responses", url))
        .json(&ConversationResponseRequest {
            model: "gpt-oss-120b".to_string(),
            input: Input::Text("I have a pet turtle named Shelly.".to_string()),
            instructions: Some("You are a helpful assistant with perfect memory.".to_string()),
            max_output_tokens: Some(100),
            temperature: Some(0.0),
            stream: Some(false),
            tools: None,
            conversation: ConversationParam {
                id: conv.id.clone(),
            },
        })
        .send()
        .await
        .expect("turn 1 failed");

    assert!(
        resp1.status().is_success(),
        "turn 1 failed: {}",
        resp1.status()
    );
    let _: Response = resp1.json().await.expect("failed to parse turn 1");
    println!("Turn 1 complete");

    // Turn 2: Add more context
    let resp2 = client
        .post(format!("{}/v1/responses", url))
        .json(&ConversationResponseRequest {
            model: "gpt-oss-120b".to_string(),
            input: Input::Text("Shelly is 5 years old.".to_string()),
            instructions: None, // Should use previous instructions from conversation
            max_output_tokens: Some(100),
            temperature: Some(0.0),
            stream: Some(false),
            tools: None,
            conversation: ConversationParam {
                id: conv.id.clone(),
            },
        })
        .send()
        .await
        .expect("turn 2 failed");

    assert!(
        resp2.status().is_success(),
        "turn 2 failed: {}",
        resp2.status()
    );
    let _: Response = resp2.json().await.expect("failed to parse turn 2");
    println!("Turn 2 complete");

    // Turn 3: Query the accumulated context
    let resp3 = client
        .post(format!("{}/v1/responses", url))
        .json(&ConversationResponseRequest {
            model: "gpt-oss-120b".to_string(),
            input: Input::Text("What is my pet's name and how old is it?".to_string()),
            instructions: None,
            max_output_tokens: Some(100),
            temperature: Some(0.0),
            stream: Some(false),
            tools: None,
            conversation: ConversationParam {
                id: conv.id.clone(),
            },
        })
        .send()
        .await
        .expect("turn 3 failed");

    assert!(
        resp3.status().is_success(),
        "turn 3 failed: {}",
        resp3.status()
    );

    let response3: Response = resp3.json().await.expect("failed to parse turn 3");
    let text = extract_final_message_text(&response3.output).expect("no message");
    let lower = text.to_lowercase();

    println!("Turn 3 response: {}", text);
    assert!(
        lower.contains("shelly"),
        "Should remember pet name 'Shelly': {}",
        text
    );
    assert!(
        lower.contains("5") || lower.contains("five"),
        "Should remember age '5': {}",
        text
    );
}

/// Test that conversation and previous_response_id cannot be used together.
#[tokio::test]
async fn test_conversation_and_previous_response_id_mutually_exclusive() {
    skip_if_no_integration!();

    let (client, url) = setup_responses_test().expect("test setup failed");

    // Create a conversation
    let conv: Conversation = client
        .post(format!("{}/v1/conversations", url))
        .json(&CreateConversationRequest {
            metadata: None,
            items: None,
        })
        .send()
        .await
        .expect("failed to create conversation")
        .json()
        .await
        .expect("failed to parse");

    // Try to use both conversation and previous_response_id
    let req = serde_json::json!({
        "model": "gpt-oss-120b",
        "input": "Hello",
        "conversation": { "id": conv.id },
        "previous_response_id": "resp_12345"
    });

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

    let error: ErrorResponse = resp.json().await.expect("failed to parse error");
    assert!(
        error.error.message.contains("Cannot use both")
            || error.error.message.contains("conversation")
            || error.error.message.contains("previous_response_id"),
        "Error should mention mutual exclusivity: {}",
        error.error.message
    );
}
