//! Core executor for Responses API requests.

use base64::Engine;
use reqwest::Client;
use rmcp::model::RawContent;
use serde_json::Value;

use crate::config::ModelConfig;
use crate::containers::ContainerStore;
use crate::mcp::{McpClient, McpError};
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CreateResponseRequest,
    FunctionToolWrapper, Response, Tool,
};
use crate::translation::{
    PendingToolCall, assistant_tool_call_message, extract_tool_calls, from_chat_completion,
    has_pending_tool_calls, to_chat_completion, tool_result_message,
};

/// Generated file from code interpreter execution.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    pub file_id: String,
    pub filename: String,
    pub container_id: String,
}

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Available models and their backend configurations
    pub models: Vec<ModelConfig>,
    /// Maximum number of tool call iterations
    pub max_tool_iterations: usize,
    /// HTTP request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
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
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("LLM error: {0}")]
    Llm(String),
}

/// The main executor that handles Responses API requests.
pub struct Executor {
    config: ExecutorConfig,
    http: Client,
    mcp: McpClient,
    containers: ContainerStore,
}

impl Executor {
    /// Create a new executor.
    ///
    /// Returns an error if the HTTP client fails to initialize (rare, usually TLS issues).
    pub fn new(
        config: ExecutorConfig,
        mcp: McpClient,
        containers: ContainerStore,
    ) -> Result<Self, reqwest::Error> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()?;

        Ok(Self {
            config,
            http,
            mcp,
            containers,
        })
    }

    /// Get a model configuration by ID.
    fn get_model(&self, model_id: &str) -> Option<&ModelConfig> {
        self.config.models.iter().find(|m| m.id == model_id)
    }

    /// Execute a Responses API request.
    ///
    /// This is the main entry point that:
    /// 1. Expands built-in tools to function definitions
    /// 2. Translates the request to Chat Completions format
    /// 3. Calls the Chat Completions backend
    /// 4. Executes any tool calls via MCP
    /// 5. Loops until completion
    /// 6. Returns the final Response with file citations
    ///
    /// # Arguments
    ///
    /// * `req` - The request to execute
    /// * `previous_messages` - Messages from resolved previous_response_id chain
    pub async fn execute(
        &self,
        req: &CreateResponseRequest,
        previous_messages: Vec<ChatMessage>,
    ) -> Result<Response, ExecutionError> {
        // Validate model exists before starting
        if self.get_model(&req.model).is_none() {
            return Err(ExecutionError::ModelNotFound(req.model.clone()));
        }

        // Expand built-in tools to function definitions
        // If no tools specified, pass empty - don't auto-inject
        let expanded_tools = self.expand_tools(&req.tools).await;

        // Check if code_interpreter is being used - if so, create a container
        let has_code_interpreter = req.tools.iter().any(|t| {
            matches!(t, Tool::Builtin(b) if b.tool_type == "code_interpreter")
        });
        let container_id = if has_code_interpreter {
            self.containers.create()
        } else {
            String::new()
        };

        // Create modified request with expanded tools
        let mut req_with_tools = req.clone();
        req_with_tools.tools = expanded_tools;

        // Initialize conversation with previous messages from chain
        let mut conversation: Vec<ChatMessage> = previous_messages;
        let mut iteration = 0;
        let mut all_generated_files: Vec<GeneratedFile> = Vec::new();

        loop {
            iteration += 1;
            tracing::info!(iteration, "Starting tool loop iteration");

            if iteration > self.config.max_tool_iterations {
                tracing::error!(
                    max = self.config.max_tool_iterations,
                    "Max tool iterations exceeded"
                );
                return Err(ExecutionError::MaxIterationsExceeded(
                    self.config.max_tool_iterations,
                ));
            }

            // Translate request to Chat Completions
            let chat_req = to_chat_completion(&req_with_tools, Some(conversation.clone()));

            // Call the backend
            tracing::info!(model = %chat_req.model, "Calling LLM");
            let chat_resp = self.call_llm(&chat_req).await?;

            // Check if we have tool calls to execute
            if has_pending_tool_calls(&chat_resp) {
                // Add assistant's tool call message to conversation
                if let Some(assistant_msg) = assistant_tool_call_message(&chat_resp) {
                    conversation.push(assistant_msg);
                }

                // Execute each tool call
                let pending_calls = extract_tool_calls(&chat_resp);
                tracing::info!(
                    count = pending_calls.len(),
                    tools = ?pending_calls.iter().map(|c| &c.name).collect::<Vec<_>>(),
                    "Executing tool calls"
                );

                for call in pending_calls {
                    tracing::info!(tool = %call.name, call_id = %call.id, "Executing tool call");
                    let (result, generated_files) =
                        self.execute_tool_call(&call, &container_id).await;

                    let result_preview: String = result.chars().take(200).collect();
                    tracing::info!(
                        tool = %call.name,
                        result_len = result.len(),
                        files_generated = generated_files.len(),
                        result_preview = %result_preview,
                        "Tool call completed"
                    );

                    all_generated_files.extend(generated_files);
                    let result_msg = tool_result_message(call.id, result);
                    conversation.push(result_msg);
                }

                // Continue loop to get model's response to tool results
                continue;
            }

            // No tool calls - we're done
            tracing::info!(
                total_files = all_generated_files.len(),
                "Request completed, no more tool calls"
            );
            return Ok(from_chat_completion(&chat_resp, req, all_generated_files));
        }
    }

    /// Expand built-in tool types to full function definitions.
    ///
    /// - `{"type": "function", ...}` passes through as-is
    /// - `{"type": "weather"}` expands to function definitions from the MCP server
    async fn expand_tools(&self, tools: &[Tool]) -> Vec<Tool> {
        let mut expanded = Vec::new();

        for tool in tools {
            match tool {
                Tool::Function(f) => {
                    // Pass through function tools as-is
                    expanded.push(Tool::Function(f.clone()));
                }
                Tool::Builtin(builtin) => {
                    // Expand built-in tool to function definitions from MCP server
                    if let Some(mcp_tools) = self
                        .mcp
                        .get_tools_by_builtin_type(&builtin.tool_type)
                        .await
                    {
                        for mcp_tool in mcp_tools {
                            expanded.push(mcp_tool_to_function_tool(mcp_tool));
                        }
                    } else {
                        tracing::warn!(
                            "Unknown built-in tool type: {}",
                            builtin.tool_type
                        );
                    }
                }
            }
        }

        expanded
    }

    /// Call the Chat Completions endpoint.
    async fn call_llm(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ExecutionError> {
        // Look up model configuration
        let model_config = self
            .get_model(&req.model)
            .ok_or_else(|| ExecutionError::ModelNotFound(req.model.clone()))?;

        let url = format!("{}/v1/chat/completions", model_config.url);

        // Build request with optional auth
        let mut request = self.http.post(&url).json(req);
        if let Some(api_key) = &model_config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

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

    /// Execute a tool call and return the text result plus any generated files.
    async fn execute_tool_call(
        &self,
        call: &PendingToolCall,
        container_id: &str,
    ) -> (String, Vec<GeneratedFile>) {
        let arguments: Value = serde_json::from_str(&call.arguments).unwrap_or(Value::Null);

        // Check if this is a code interpreter call (execute_python)
        let is_code_interpreter = call.name == "execute_python";

        match self.mcp.call_tool(&call.name, arguments).await {
            Ok(result) => {
                let mut text_parts = Vec::new();
                let mut generated_files = Vec::new();

                for content in &result.content {
                    match &content.raw {
                        RawContent::Text(tc) => {
                            text_parts.push(tc.text.as_str());
                        }
                        RawContent::Image(img) => {
                            if is_code_interpreter {
                                // Store the image in the container
                                if let Some(file_id) = self.store_image(
                                    container_id,
                                    &img.data,
                                    &img.mime_type,
                                    generated_files.len(),
                                ) {
                                    let filename = format!("output_{}.png", generated_files.len());
                                    generated_files.push(GeneratedFile {
                                        file_id,
                                        filename,
                                        container_id: container_id.to_string(),
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }

                let text = text_parts.join("\n");

                if result.is_error.unwrap_or(false) {
                    (format!("Error: {}", text), generated_files)
                } else {
                    (text, generated_files)
                }
            }
            Err(e) => {
                tracing::error!("Tool call {} failed: {}", call.name, e);
                (format!("Error: {}", e), vec![])
            }
        }
    }

    /// Store a base64-encoded image in the container store.
    fn store_image(
        &self,
        container_id: &str,
        base64_data: &str,
        mime_type: &str,
        index: usize,
    ) -> Option<String> {
        // Decode base64 image data
        let content = base64::engine::general_purpose::STANDARD
            .decode(base64_data)
            .ok()?;

        // Determine filename from MIME type
        let ext = match mime_type {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/gif" => "gif",
            "image/webp" => "webp",
            _ => "bin",
        };
        let filename = format!("output_{}.{}", index, ext);

        // Store in container
        let file_id = self.containers.add_file(container_id, filename, content, mime_type)?;

        tracing::debug!(
            "Stored code interpreter output: {} in container {}",
            file_id,
            container_id
        );

        Some(file_id)
    }
}

/// Convert an MCP tool to a Responses API function tool.
fn mcp_tool_to_function_tool(mcp_tool: rmcp::model::Tool) -> Tool {
    // input_schema is Arc<Map<String, Value>> - convert to Value
    let parameters = serde_json::to_value(&*mcp_tool.input_schema).ok();

    Tool::Function(FunctionToolWrapper {
        tool_type: "function".to_string(),
        name: mcp_tool.name.to_string(),
        description: mcp_tool.description.map(|d| d.to_string()),
        parameters,
        strict: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_tool_iterations, 10);
        assert_eq!(config.timeout_secs, 300);
        assert!(config.models.is_empty());
    }

    #[test]
    fn get_model_lookup() {
        let config = ExecutorConfig {
            models: vec![
                ModelConfig::new("model-a", "http://a:8000"),
                ModelConfig::new("model-b", "http://b:8000"),
            ],
            ..Default::default()
        };
        let mcp = McpClient::new(vec![]);
        let containers = ContainerStore::new();
        let executor = Executor::new(config, mcp, containers).expect("create executor");

        assert!(executor.get_model("model-a").is_some());
        assert!(executor.get_model("model-b").is_some());
        assert!(executor.get_model("model-c").is_none());
    }
}
