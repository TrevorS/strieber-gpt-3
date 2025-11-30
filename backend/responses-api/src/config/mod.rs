//! Configuration for the Responses API service.
//!
//! Supports configuration via:
//! - Environment variables (MCP_CONFIG, CHAT_COMPLETIONS_URL, etc.)
//! - Programmatic configuration

use serde::Deserialize;
use std::env;
use std::time::Duration;

use crate::mcp::McpServerConfig;

/// Main configuration for the Responses API service.
#[derive(Debug, Clone)]
pub struct Config {
    /// URL of the Chat Completions API backend (any OpenAI-compatible inference engine)
    pub chat_completions_url: String,
    /// MCP server configurations
    pub mcp_servers: Vec<McpServerConfig>,
    /// Maximum tool calling iterations
    pub max_tool_iterations: usize,
    /// HTTP request timeout
    pub timeout: Duration,
    /// Response TTL (time to live in storage)
    pub response_ttl: Duration,
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chat_completions_url: "http://localhost:8000".to_string(),
            mcp_servers: Vec::new(),
            max_tool_iterations: 10,
            timeout: Duration::from_secs(300),
            response_ttl: Duration::from_secs(3600),
            port: 8000,
            host: "0.0.0.0".to_string(),
        }
    }
}

/// JSON structure for MCP_CONFIG environment variable.
#[derive(Debug, Deserialize)]
struct McpConfigJson {
    servers: Vec<McpServerConfig>,
}

impl Config {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `CHAT_COMPLETIONS_URL`: URL of the Chat Completions API backend (default: http://localhost:8000)
    /// - `MCP_CONFIG`: JSON object with MCP server configuration
    ///   Example: {"servers":[{"name":"weather","url":"http://mcp-weather:8000/mcp"}]}
    /// - `MAX_TOOL_ITERATIONS`: Maximum tool calling iterations (default: 10)
    /// - `TIMEOUT_SECS`: HTTP request timeout in seconds (default: 300)
    /// - `RESPONSE_TTL_SECS`: Response TTL in seconds (default: 3600)
    /// - `PORT`: Port to listen on (default: 8000)
    /// - `HOST`: Host to bind to (default: 0.0.0.0)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = env::var("CHAT_COMPLETIONS_URL") {
            config.chat_completions_url = url;
        }

        // Parse MCP configuration from JSON env var
        if let Ok(json) = env::var("MCP_CONFIG") {
            match serde_json::from_str::<McpConfigJson>(&json) {
                Ok(mcp_config) => config.mcp_servers = mcp_config.servers,
                Err(e) => tracing::error!("Failed to parse MCP_CONFIG: {}", e),
            }
        }

        if let Ok(v) = env::var("MAX_TOOL_ITERATIONS")
            && let Ok(n) = v.parse()
        {
            config.max_tool_iterations = n;
        }

        if let Ok(v) = env::var("TIMEOUT_SECS")
            && let Ok(secs) = v.parse()
        {
            config.timeout = Duration::from_secs(secs);
        }

        if let Ok(v) = env::var("RESPONSE_TTL_SECS")
            && let Ok(secs) = v.parse()
        {
            config.response_ttl = Duration::from_secs(secs);
        }

        if let Ok(v) = env::var("PORT")
            && let Ok(p) = v.parse()
        {
            config.port = p;
        }

        if let Ok(host) = env::var("HOST") {
            config.host = host;
        }

        config
    }

    /// Add an MCP server to the configuration.
    pub fn add_mcp_server(mut self, name: &str, url: &str) -> Self {
        self.mcp_servers.push(McpServerConfig::new(name, url));
        self
    }

    /// Add an MCP server with a tool name prefix.
    pub fn add_mcp_server_with_prefix(mut self, name: &str, url: &str, prefix: &str) -> Self {
        self.mcp_servers
            .push(McpServerConfig::new(name, url).with_prefix(prefix));
        self
    }

    /// Set the Chat Completions API URL.
    pub fn chat_completions_url(mut self, url: &str) -> Self {
        self.chat_completions_url = url.to_string();
        self
    }

    /// Set the listening port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the max tool iterations.
    pub fn max_tool_iterations(mut self, max: usize) -> Self {
        self.max_tool_iterations = max;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let config = Config::default();
        assert!(config.chat_completions_url.contains("localhost"));
        assert_eq!(config.port, 8000);
        assert_eq!(config.max_tool_iterations, 10);
    }

    #[test]
    fn builder_pattern_works() {
        let config = Config::default()
            .chat_completions_url("http://custom:9000")
            .port(3000)
            .max_tool_iterations(5)
            .add_mcp_server("test", "http://test:8000/mcp");

        assert_eq!(config.chat_completions_url, "http://custom:9000");
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_tool_iterations, 5);
        assert_eq!(config.mcp_servers.len(), 1);
    }

    #[test]
    fn parse_mcp_config_json() {
        let json = r#"{"servers":[{"name":"weather","url":"http://mcp-weather:8000/mcp"}]}"#;
        let config: McpConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "weather");
        assert_eq!(config.servers[0].url, "http://mcp-weather:8000/mcp");
    }

    #[test]
    fn parse_mcp_config_with_prefix() {
        let json = r#"{"servers":[{"name":"search","url":"http://mcp-search:8000/mcp","tool_prefix":"search_"}]}"#;
        let config: McpConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "search");
        assert_eq!(
            config.servers[0].tool_prefix,
            Some("search_".to_string())
        );
    }

    #[test]
    fn parse_mcp_config_multiple_servers() {
        let json = r#"{"servers":[{"name":"weather","url":"http://mcp-weather:8000/mcp"},{"name":"code","url":"http://mcp-code:8000/mcp"}]}"#;
        let config: McpConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.servers.len(), 2);
        assert_eq!(config.servers[0].name, "weather");
        assert_eq!(config.servers[1].name, "code");
    }
}
