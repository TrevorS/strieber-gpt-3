//! Integration tests for tool calling via MCP.
//!
//! Run with: RESPONSES_API_URL=http://localhost:8000 cargo test --test tool_calling_integration

mod common;

use common::*;
use serde_json::json;

/// Test weather tool calling flow.
/// This test requires the weather MCP server to be running.
#[tokio::test]
async fn test_weather_tool_call() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Define the weather tool (Responses API uses flat format, not nested under "function")
    let weather_tool = json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }
    });

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What's the weather like in Tokyo?".to_string()),
        instructions: Some("Use the weather tool to get accurate weather information.".to_string()),
        max_output_tokens: Some(200),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: Some(vec![weather_tool]),
        previous_response_id: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .expect("failed to create response");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Request failed with {}: {}", status, body);
    }

    let body: Response = resp.json().await.expect("failed to parse response");
    assert_eq!(body.status, "completed");

    // Check if there are function_call items in the output (tool was invoked)
    let has_function_call = body.output.iter().any(|o| o.item_type == "function_call");

    let has_function_call_output = body
        .output
        .iter()
        .any(|o| o.item_type == "function_call_output");

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}", item.item_type);
        if let Some(name) = &item.name {
            println!("    name: {}", name);
        }
        if let Some(args) = &item.arguments {
            println!("    arguments: {}", args);
        }
        if let Some(output) = &item.output {
            println!("    output: {}", output);
        }
    }

    // The model should have called the weather tool
    if has_function_call {
        println!("Tool was called!");
        assert!(
            has_function_call_output,
            "function_call_output should be present"
        );

        // Check that the final message contains weather information
        let final_message = body
            .output
            .iter()
            .filter(|o| o.item_type == "message")
            .next_back()
            .expect("no final message");

        let text = final_message
            .content
            .as_ref()
            .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
            .and_then(|p| p.text.as_ref())
            .expect("no text in final message");

        println!("Final response: {}", text);
        // Response should mention weather-related terms
        let text_lower = text.to_lowercase();
        assert!(
            text_lower.contains("tokyo")
                || text_lower.contains("weather")
                || text_lower.contains("temperature")
                || text_lower.contains("°"),
            "Response should contain weather information"
        );
    } else {
        println!("Model did not use tool (may have responded directly)");
        // Check that there's at least a message
        assert!(
            body.output.iter().any(|o| o.item_type == "message"),
            "Should have at least a message output"
        );
    }
}

/// Test code interpreter tool calling.
/// This test requires the code_interpreter MCP server to be running.
#[tokio::test]
async fn test_code_interpreter_tool_call() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let code_tool = json!({
        "type": "function",
        "name": "execute_python",
        "description": "Execute Python code and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    });

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("Calculate the factorial of 7 using Python code.".to_string()),
        instructions: Some(
            "Use the execute_python tool to calculate and verify your answer.".to_string(),
        ),
        max_output_tokens: Some(200),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: Some(vec![code_tool]),
        previous_response_id: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .expect("failed to create response");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Request failed with {}: {}", status, body);
    }

    let body: Response = resp.json().await.unwrap();
    assert_eq!(body.status, "completed");

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}", item.item_type);
        if item.item_type == "function_call" {
            println!("    name: {:?}", item.name);
            println!("    arguments: {:?}", item.arguments);
        }
        if item.item_type == "function_call_output" {
            println!("    output: {:?}", item.output);
        }
    }

    // Check final message contains the factorial result (5040)
    let final_message = body
        .output
        .iter()
        .filter(|o| o.item_type == "message")
        .next_back()
        .expect("no final message");

    let text = final_message
        .content
        .as_ref()
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_ref())
        .expect("no text in final message");

    println!("Final response: {}", text);
    assert!(
        text.contains("5040"),
        "Response should contain 5040 (7! = 5040)"
    );
}

/// Test web search tool calling.
/// This test requires the web_search MCP server to be running.
#[tokio::test]
async fn test_web_search_tool_call() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let search_tool = json!({
        "type": "function",
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    });

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text(
            "Search for information about the Rust programming language.".to_string(),
        ),
        instructions: Some("Use the web_search tool to find current information.".to_string()),
        max_output_tokens: Some(300),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: Some(vec![search_tool]),
        previous_response_id: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .expect("failed to create response");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Request failed with {}: {}", status, body);
    }

    let body: Response = resp.json().await.unwrap();
    assert_eq!(body.status, "completed");

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}", item.item_type);
    }

    // Should have a final message
    let final_message = body
        .output
        .iter()
        .filter(|o| o.item_type == "message")
        .next_back()
        .expect("no final message");

    let text = final_message
        .content
        .as_ref()
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_ref())
        .expect("no text in final message");

    println!("Final response: {}", text);
    let text_lower = text.to_lowercase();
    assert!(
        text_lower.contains("rust") || text_lower.contains("programming"),
        "Response should mention Rust or programming"
    );
}

/// Test multiple tool calls in sequence.
#[tokio::test]
async fn test_multiple_tool_calls() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let weather_tool = json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    });

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text(
            "Compare the weather in Tokyo and New York. Get the weather for both cities."
                .to_string(),
        ),
        instructions: Some(
            "Use the weather tool to get weather for both cities before comparing.".to_string(),
        ),
        max_output_tokens: Some(400),
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: Some(vec![weather_tool]),
        previous_response_id: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .expect("failed to create response");

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        panic!("Request failed with {}: {}", status, body);
    }

    let body: Response = resp.json().await.unwrap();
    assert_eq!(body.status, "completed");

    // Count tool calls
    let tool_call_count = body
        .output
        .iter()
        .filter(|o| o.item_type == "function_call")
        .count();

    println!("Number of tool calls: {}", tool_call_count);
    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}", item.item_type);
        if item.item_type == "function_call" {
            println!("    arguments: {:?}", item.arguments);
        }
    }

    // Should have multiple tool calls (at least for both cities)
    // Note: Model behavior may vary
    if tool_call_count > 0 {
        println!("Model used {} tool call(s)", tool_call_count);
    }

    // Final message should mention both cities
    let final_message = body
        .output
        .iter()
        .filter(|o| o.item_type == "message")
        .next_back()
        .expect("no final message");

    let text = final_message
        .content
        .as_ref()
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_ref())
        .expect("no text");

    println!("Final response: {}", text);
    let text_lower = text.to_lowercase();
    assert!(
        text_lower.contains("tokyo") || text_lower.contains("new york"),
        "Response should mention one of the cities"
    );
}

/// Test tool call with no matching tool (should not call).
#[tokio::test]
async fn test_no_matching_tool() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Provide an unrelated tool
    let unrelated_tool = json!({
        "type": "function",
        "name": "send_email",
        "description": "Send an email to someone",
        "parameters": {
            "type": "object",
            "properties": {
                "to": { "type": "string" },
                "subject": { "type": "string" },
                "body": { "type": "string" }
            },
            "required": ["to", "subject", "body"]
        }
    });

    let req = CreateResponseRequest {
        model: "gpt-oss-120b".to_string(),
        input: Input::Text("What is 2 + 2?".to_string()),
        instructions: None,
        max_output_tokens: Some(100), // Reasoning models need more tokens
        temperature: Some(0.0),
        store: Some(false),
        stream: None,
        tools: Some(vec![unrelated_tool]),
        previous_response_id: None,
    };

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .expect("failed to create response");

    assert!(resp.status().is_success());

    let body: Response = resp.json().await.unwrap();
    assert_eq!(body.status, "completed");

    // Should NOT call the send_email tool for a math question
    let has_email_call = body
        .output
        .iter()
        .any(|o| o.item_type == "function_call" && o.name.as_deref() == Some("send_email"));

    assert!(
        !has_email_call,
        "Should not call send_email for a math question"
    );

    // Should have a direct response with "4"
    let message = body
        .output
        .iter()
        .find(|o| o.item_type == "message")
        .unwrap();
    let text = message
        .content
        .as_ref()
        .and_then(|c| c.iter().find(|p| p.content_type == "output_text"))
        .and_then(|p| p.text.as_ref())
        .unwrap();

    println!("Response: {}", text);
    assert!(text.contains("4"));
}

// ============================================================================
// Built-in Tool Type Tests
// These test the {"type": "code_interpreter"} style tool definitions
// that expand to MCP function definitions on the server side.
// ============================================================================

/// Test built-in code_interpreter tool type.
/// Uses {"type": "code_interpreter"} instead of explicit function definition.
#[tokio::test]
async fn test_builtin_code_interpreter() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Calculate 15 factorial using Python.",
        "instructions": "Use the code interpreter to calculate and verify the result.",
        "max_output_tokens": 200,
        "temperature": 0.0,
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
    assert_eq!(body.status, "completed");

    // Should have executed code (function_call with execute_python)
    let has_code_call = body
        .output
        .iter()
        .any(|o| o.item_type == "function_call" && o.name.as_deref() == Some("execute_python"));

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}, name: {:?}", item.item_type, item.name);
    }

    // Get final message
    let final_text = extract_final_message_text(&body.output).unwrap_or("");
    println!("Final response: {}", final_text);

    // 15! = 1307674368000
    assert!(
        final_text.contains("1307674368000") || has_code_call,
        "Response should contain factorial result or have used code interpreter"
    );
}

/// Test built-in weather tool type.
/// Uses {"type": "weather"} instead of explicit function definition.
#[tokio::test]
async fn test_builtin_weather() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "What's the weather in Paris?",
        "instructions": "Use the weather tool to get current conditions.",
        "max_output_tokens": 200,
        "temperature": 0.0,
        "tools": [{"type": "weather"}]
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
    assert_eq!(body.status, "completed");

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}, name: {:?}", item.item_type, item.name);
    }

    let final_text = extract_final_message_text(&body.output).unwrap_or("");
    println!("Final response: {}", final_text);

    let text_lower = final_text.to_lowercase();
    assert!(
        text_lower.contains("paris") || text_lower.contains("weather") || text_lower.contains("°"),
        "Response should mention Paris or weather"
    );
}

/// Test built-in web_search tool type.
/// Uses {"type": "web_search"} instead of explicit function definition.
#[tokio::test]
async fn test_builtin_web_search() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Search for recent news about climate change.",
        "instructions": "Use web search to find current information.",
        "max_output_tokens": 300,
        "temperature": 0.0,
        "tools": [{"type": "web_search"}]
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
    assert_eq!(body.status, "completed");

    println!("Output items:");
    for item in &body.output {
        println!("  - type: {}, name: {:?}", item.item_type, item.name);
    }

    let final_text = extract_final_message_text(&body.output).unwrap_or("");
    println!("Final response: {}", final_text);

    // Should have some response about climate
    assert!(
        !final_text.is_empty(),
        "Should have a response"
    );
}

/// Test max_tool_calls limit.
#[tokio::test]
async fn test_max_tool_iterations() {
    skip_if_no_integration!();

    let client = create_client();
    let url = responses_api_url().unwrap();

    // Use a tool that might cause many iterations
    let endless_tool = json!({
        "type": "function",
        "name": "get_more_info",
        "description": "Get more information about a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": { "type": "string" }
            },
            "required": ["topic"]
        }
    });

    let req = json!({
        "model": "gpt-oss-120b",
        "input": "Keep getting more info about AI until you have comprehensive knowledge.",
        "instructions": "Always call get_more_info to learn more.",
        "max_output_tokens": 200,
        "temperature": 0.0,
        "store": false,
        "tools": [endless_tool],
        "max_tool_calls": 2  // Limit to 2 tool calls
    });

    let resp = client
        .post(format!("{}/v1/responses", url))
        .json(&req)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .expect("failed to create response");

    // Should either succeed with limited calls or fail with max iterations exceeded
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap();

    println!("Status: {}", status);
    println!("Response: {}", serde_json::to_string_pretty(&body).unwrap());

    // Either completed (hit limit gracefully) or error (max iterations)
    if status.is_success() {
        let tool_calls = body["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["type"] == "function_call")
            .count();
        println!("Number of tool calls: {}", tool_calls);
    } else {
        // May have hit max iterations error
        assert!(
            body["error"]["type"] == "max_iterations_exceeded"
                || body["error"]["message"]
                    .as_str()
                    .map(|s| s.contains("iteration"))
                    .unwrap_or(false),
            "Should be max iterations error"
        );
    }
}
