//! Core executor for Responses API requests.

use reqwest::Client;
use serde_json::Value;

use crate::mcp::{McpClient, McpError};
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CreateResponseRequest, Response,
};
use crate::translation::{
    PendingToolCall, assistant_tool_call_message, extract_tool_calls, from_chat_completion,
    has_pending_tool_calls, to_chat_completion, tool_result_message,
};

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// URL of the Chat Completions API backend
    pub chat_completions_url: String,
    /// Maximum number of tool call iterations
    pub max_tool_iterations: usize,
    /// HTTP request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            chat_completions_url: "http://localhost:8000".to_string(),
            max_tool_iterations: 10,
            timeout_secs: 300,
        }
    }
}

/// Errors that can occur during execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Max tool iterations ({0}) exceeded")]
    MaxIterationsExceeded(usize),
    #[error("LLM error: {0}")]
    Llm(String),
}

/// The main executor that handles Responses API requests.
pub struct Executor {
    config: ExecutorConfig,
    http: Client,
    mcp: McpClient,
}

impl Executor {
    /// Create a new executor.
    pub fn new(config: ExecutorConfig, mcp: McpClient) -> Self {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to create HTTP client");

        Self { config, http, mcp }
    }

    /// Execute a Responses API request.
    ///
    /// This is the main entry point that:
    /// 1. Translates the request to Chat Completions format
    /// 2. Calls the Chat Completions backend
    /// 3. Executes any tool calls via MCP
    /// 4. Loops until completion
    /// 5. Returns the final Response
    pub async fn execute(&self, req: &CreateResponseRequest) -> Result<Response, ExecutionError> {
        let mut conversation: Vec<ChatMessage> = Vec::new();
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > self.config.max_tool_iterations {
                return Err(ExecutionError::MaxIterationsExceeded(
                    self.config.max_tool_iterations,
                ));
            }

            // Translate request to Chat Completions
            let chat_req = to_chat_completion(req, Some(conversation.clone()));

            // Call the backend
            let chat_resp = self.call_llm(&chat_req).await?;

            // Check if we have tool calls to execute
            if has_pending_tool_calls(&chat_resp) {
                // Add assistant's tool call message to conversation
                if let Some(assistant_msg) = assistant_tool_call_message(&chat_resp) {
                    conversation.push(assistant_msg);
                }

                // Execute each tool call
                let pending_calls = extract_tool_calls(&chat_resp);
                for call in pending_calls {
                    let result = self.execute_tool_call(&call).await;
                    let result_msg = tool_result_message(call.id, result);
                    conversation.push(result_msg);
                }

                // Continue loop to get model's response to tool results
                continue;
            }

            // No tool calls - we're done
            return Ok(from_chat_completion(&chat_resp, req));
        }
    }

    /// Call the Chat Completions endpoint.
    async fn call_llm(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ExecutionError> {
        let url = format!("{}/v1/chat/completions", self.config.chat_completions_url);

        let response = self.http.post(&url).json(req).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ExecutionError::Llm(format!(
                "LLM returned {}: {}",
                status, body
            )));
        }

        let chat_resp: ChatCompletionResponse = response.json().await?;
        Ok(chat_resp)
    }

    async fn execute_tool_call(&self, call: &PendingToolCall) -> String {
        let arguments: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);

        match self.mcp.call_tool_text(&call.name, arguments).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Tool call {} failed: {}", call.name, e);
                format!("Error: {}", e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_tool_iterations, 10);
        assert_eq!(config.timeout_secs, 300);
        assert!(config.chat_completions_url.contains("localhost"));
    }
}
