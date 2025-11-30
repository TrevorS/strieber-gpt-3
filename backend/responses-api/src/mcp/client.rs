//! MCP client using the official rmcp SDK with Streamable HTTP transport.

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{CallToolRequestParam, CallToolResult, ListToolsResult, RawContent};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;
use tokio::sync::RwLock;
use std::borrow::Cow;

/// Configuration for an MCP server endpoint.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Server name (for identification)
    pub name: String,
    /// Base URL (e.g., "http://mcp-weather:8000/mcp")
    pub url: String,
    /// Tool name prefix to add (e.g., "weather_" for disambiguation)
    pub tool_prefix: Option<String>,
}

impl McpServerConfig {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            tool_prefix: None,
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
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
pub struct McpClient {
    /// Server configurations
    configs: Vec<McpServerConfig>,
    /// Connected servers (lazily populated)
    connections: Arc<RwLock<HashMap<String, ConnectedServer>>>,
    /// Tool routing: tool_name → server_name
    tool_routing: Arc<RwLock<HashMap<String, String>>>,
}

impl McpClient {
    /// Create a new MCP client with server configurations.
    pub fn new(configs: Vec<McpServerConfig>) -> Self {
        Self {
            configs,
            connections: Arc::new(RwLock::new(HashMap::new())),
            tool_routing: Arc::new(RwLock::new(HashMap::new())),
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
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError> {
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
        let actual_name: Cow<'static, str> = if let Some(prefix) = &server.config.tool_prefix {
            name.strip_prefix(prefix).unwrap_or(name).to_string().into()
        } else {
            name.to_string().into()
        };

        // Call the tool
        let params = CallToolRequestParam {
            name: actual_name,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_config_with_prefix() {
        let config = McpServerConfig::new("weather", "http://mcp-weather:8000/mcp")
            .with_prefix("weather_");
        assert_eq!(config.name, "weather");
        assert_eq!(config.url, "http://mcp-weather:8000/mcp");
        assert_eq!(config.tool_prefix, Some("weather_".to_string()));
    }
}
