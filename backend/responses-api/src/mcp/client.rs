//! MCP client using the official rmcp SDK with Streamable HTTP transport.

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{CallToolRequestParam, CallToolResult, ListToolsResult, RawContent};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

/// Configuration for an MCP server endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name (for identification)
    pub name: String,
    /// Base URL (e.g., "http://mcp-weather:8000/mcp")
    pub url: String,
    /// Tool name prefix to add (e.g., "weather_" for disambiguation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_prefix: Option<String>,
    /// Built-in tool type (e.g., "weather", "web_search", "code_interpreter")
    /// Allows clients to enable this server's tools with `{"type": "weather"}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub builtin_type: Option<String>,
}

impl McpServerConfig {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            tool_prefix: None,
            builtin_type: None,
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    pub fn with_builtin_type(mut self, builtin_type: impl Into<String>) -> Self {
        self.builtin_type = Some(builtin_type.into());
        self
    }
}

/// Error type for MCP operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Server not found: {0}")]
    ServerNotFound(String),
    #[error("Tool execution failed: {0}")]
    ToolExecution(String),
}

/// A connected MCP server with its tools.
struct ConnectedServer {
    /// The running service (connection to the MCP server)
    _service: RunningService<RoleClient, ()>,
    /// Peer handle for making calls
    peer: rmcp::service::Peer<RoleClient>,
    /// Config for this server
    config: McpServerConfig,
}

/// MCP client that manages connections to multiple MCP servers.
#[derive(Clone)]
pub struct McpClient {
    configs: Vec<McpServerConfig>,
    connections: Arc<RwLock<HashMap<String, ConnectedServer>>>,
    tool_routing: Arc<RwLock<HashMap<String, String>>>,
    /// Maps builtin_type (e.g., "weather") to server name
    builtin_type_routing: Arc<RwLock<HashMap<String, String>>>,
}

impl McpClient {
    /// Create a new MCP client with server configurations.
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        // Build builtin_type routing from configs
        let mut builtin_routing = HashMap::new();
        for config in &configs {
            if let Some(builtin_type) = &config.builtin_type {
                builtin_routing.insert(builtin_type.clone(), config.name.clone());
            }
        }

        Self {
            configs,
            connections: Arc::new(RwLock::new(HashMap::new())),
            tool_routing: Arc::new(RwLock::new(HashMap::new())),
            builtin_type_routing: Arc::new(RwLock::new(builtin_routing)),
        }
    }

    /// Connect to all configured servers and discover their tools.
    pub async fn connect_all(&self) -> Result<(), McpError> {
        for config in &self.configs {
            if let Err(e) = self.connect_server(config.clone()).await {
                tracing::warn!("Failed to connect to MCP server {}: {}", config.name, e);
            }
        }
        Ok(())
    }

    /// Connect to a specific MCP server.
    async fn connect_server(&self, config: McpServerConfig) -> Result<(), McpError> {
        let transport = StreamableHttpClientTransport::from_uri(config.url.clone());

        // Connect using rmcp SDK
        let service = ()
            .serve(transport)
            .await
            .map_err(|e| McpError::Connection(format!("{}: {}", config.name, e)))?;

        let peer = service.peer().clone();

        // Discover tools from this server
        let tools_result: ListToolsResult = peer
            .list_tools(Default::default())
            .await
            .map_err(|e| McpError::Connection(format!("list_tools failed: {}", e)))?;

        // Register tool routing
        let mut routing = self.tool_routing.write().await;
        for tool in &tools_result.tools {
            let tool_name = if let Some(prefix) = &config.tool_prefix {
                format!("{}{}", prefix, tool.name)
            } else {
                tool.name.to_string()
            };
            routing.insert(tool_name, config.name.clone());
            tracing::debug!("Registered tool: {} from server {}", tool.name, config.name);
        }

        // Store connection
        let mut connections = self.connections.write().await;
        connections.insert(
            config.name.clone(),
            ConnectedServer {
                _service: service,
                peer,
                config,
            },
        );

        Ok(())
    }

    /// List all available tools from all connected servers.
    pub async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        let mut all_tools = Vec::new();
        let connections = self.connections.read().await;

        for (_, server) in connections.iter() {
            let result: ListToolsResult = server
                .peer
                .list_tools(Default::default())
                .await
                .map_err(|e| McpError::Connection(format!("list_tools failed: {}", e)))?;

            for mut tool in result.tools {
                // Apply prefix if configured
                if let Some(prefix) = &server.config.tool_prefix {
                    tool.name = format!("{}{}", prefix, tool.name).into();
                }
                all_tools.push(tool);
            }
        }

        Ok(all_tools)
    }

    /// Call a tool by name.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<CallToolResult, McpError> {
        // Find which server handles this tool
        let routing = self.tool_routing.read().await;
        let server_name = routing
            .get(name)
            .ok_or_else(|| McpError::ToolNotFound(name.to_string()))?
            .clone();
        drop(routing);

        let connections = self.connections.read().await;
        let server = connections
            .get(&server_name)
            .ok_or_else(|| McpError::ServerNotFound(server_name.clone()))?;

        // Strip prefix from tool name when calling
        let actual_name = if let Some(prefix) = &server.config.tool_prefix {
            name.strip_prefix(prefix).unwrap_or(name).to_string()
        } else {
            name.to_string()
        };

        // Call the tool
        let params = CallToolRequestParam {
            name: actual_name.into(),
            arguments: if arguments.is_null() {
                None
            } else {
                Some(arguments.as_object().cloned().unwrap_or_default())
            },
        };

        let result = server
            .peer
            .call_tool(params)
            .await
            .map_err(|e| McpError::ToolExecution(format!("{}", e)))?;

        Ok(result)
    }

    /// Call a tool and extract text result as a string.
    pub async fn call_tool_text(&self, name: &str, arguments: Value) -> Result<String, McpError> {
        let result = self.call_tool(name, arguments).await?;

        // Extract text content from result
        let text = result
            .content
            .iter()
            .filter_map(|c| {
                if let RawContent::Text(tc) = &c.raw {
                    Some(tc.text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error.unwrap_or(false) {
            Err(McpError::ToolExecution(text))
        } else {
            Ok(text)
        }
    }

    /// Check if a tool is available.
    pub async fn has_tool(&self, name: &str) -> bool {
        let routing = self.tool_routing.read().await;
        routing.contains_key(name)
    }

    /// Get list of available tool names.
    pub async fn available_tools(&self) -> Vec<String> {
        let routing = self.tool_routing.read().await;
        routing.keys().cloned().collect()
    }

    /// Get tools from the MCP server with the given builtin_type.
    /// Returns None if no server is registered with that builtin_type.
    pub async fn get_tools_by_builtin_type(
        &self,
        builtin_type: &str,
    ) -> Option<Vec<rmcp::model::Tool>> {
        // Find server name for this builtin_type
        let routing = self.builtin_type_routing.read().await;
        tracing::debug!(
            "Looking up builtin_type={}, routing keys: {:?}",
            builtin_type,
            routing.keys().collect::<Vec<_>>()
        );
        let server_name = routing.get(builtin_type)?.clone();
        drop(routing);

        // Get the server connection
        let connections = self.connections.read().await;
        let server = connections.get(&server_name)?;

        // Get tools from this server
        let result: ListToolsResult = server.peer.list_tools(Default::default()).await.ok()?;

        // Apply prefix if configured
        let tools: Vec<rmcp::model::Tool> = result
            .tools
            .into_iter()
            .map(|mut tool| {
                if let Some(prefix) = &server.config.tool_prefix {
                    tool.name = format!("{}{}", prefix, tool.name).into();
                }
                tool
            })
            .collect();

        Some(tools)
    }

    /// Get list of available builtin tool types.
    pub async fn available_builtin_types(&self) -> Vec<String> {
        let routing = self.builtin_type_routing.read().await;
        routing.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_config_with_prefix() {
        let config =
            McpServerConfig::new("weather", "http://mcp-weather:8000/mcp").with_prefix("weather_");
        assert_eq!(config.name, "weather");
        assert_eq!(config.url, "http://mcp-weather:8000/mcp");
        assert_eq!(config.tool_prefix, Some("weather_".to_string()));
    }

    #[test]
    fn server_config_with_builtin_type() {
        let config = McpServerConfig::new("code_interpreter", "http://mcp:8000/mcp")
            .with_builtin_type("code_interpreter");
        assert_eq!(config.name, "code_interpreter");
        assert_eq!(config.builtin_type, Some("code_interpreter".to_string()));
    }

    #[test]
    fn builtin_type_routing_initialized_from_config() {
        let configs = vec![
            McpServerConfig::new("weather-server", "http://mcp-weather:8000/mcp")
                .with_builtin_type("weather"),
            McpServerConfig::new("code-server", "http://mcp-code:8000/mcp")
                .with_builtin_type("code_interpreter"),
            McpServerConfig::new("generic-server", "http://mcp-generic:8000/mcp"),
        ];

        let client = McpClient::new(configs);

        // Access the builtin_type_routing directly via blocking read
        let rt = tokio::runtime::Runtime::new().unwrap();
        let routing = rt.block_on(async {
            let routing = client.builtin_type_routing.read().await;
            routing.clone()
        });

        assert_eq!(routing.get("weather"), Some(&"weather-server".to_string()));
        assert_eq!(
            routing.get("code_interpreter"),
            Some(&"code-server".to_string())
        );
        assert!(!routing.contains_key("nonexistent"));
        // generic-server has no builtin_type so shouldn't be in routing
        assert!(!routing.values().any(|v| v == "generic-server"));
    }

    #[tokio::test]
    async fn available_builtin_types() {
        let configs = vec![
            McpServerConfig::new("weather", "http://mcp:8000/mcp").with_builtin_type("weather"),
            McpServerConfig::new("code", "http://mcp:8000/mcp")
                .with_builtin_type("code_interpreter"),
        ];
        let client = McpClient::new(configs);

        let types = client.available_builtin_types().await;
        assert!(types.contains(&"weather".to_string()));
        assert!(types.contains(&"code_interpreter".to_string()));
        assert_eq!(types.len(), 2);
    }

    #[tokio::test]
    async fn get_tools_by_unknown_builtin_type_returns_none() {
        let configs = vec![
            McpServerConfig::new("weather", "http://mcp:8000/mcp").with_builtin_type("weather"),
        ];
        let client = McpClient::new(configs);

        // Unknown type should return None (no panic)
        let result = client.get_tools_by_builtin_type("nonexistent").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn empty_client_has_no_tools() {
        let client = McpClient::new(vec![]);

        assert!(client.available_tools().await.is_empty());
        assert!(client.available_builtin_types().await.is_empty());
        assert!(!client.has_tool("anything").await);
    }
}
