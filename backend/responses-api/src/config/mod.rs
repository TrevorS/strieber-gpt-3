//! Configuration for the Responses API service.
//!
//! Supports configuration via:
//! - Environment variables (MCP_SERVERS, LLAMA_URL, etc.)
//! - Programmatic configuration

use std::env;
use std::time::Duration;

use crate::mcp::McpServerConfig;

/// Main configuration for the Responses API service.
#[derive(Debug, Clone)]
pub struct Config {
    /// URL of the llama.cpp server
    pub llama_url: String,
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
            llama_url: "http://llama-server:8000".to_string(),
            mcp_servers: Vec::new(),
            max_tool_iterations: 10,
            timeout: Duration::from_secs(300),
            response_ttl: Duration::from_secs(3600),
            port: 8000,
            host: "0.0.0.0".to_string(),
        }
    }
}

impl Config {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `LLAMA_URL`: URL of the llama.cpp server (default: http://llama-server:8000)
    /// - `MCP_SERVERS`: Comma-separated list of MCP servers in format "name:url" or "name:url:prefix"
    ///   Example: "weather:http://mcp-weather:8000/mcp,search:http://mcp-search:8000/mcp:search_"
    /// - `MAX_TOOL_ITERATIONS`: Maximum tool calling iterations (default: 10)
    /// - `TIMEOUT_SECS`: HTTP request timeout in seconds (default: 300)
    /// - `RESPONSE_TTL_SECS`: Response TTL in seconds (default: 3600)
    /// - `PORT`: Port to listen on (default: 8000)
    /// - `HOST`: Host to bind to (default: 0.0.0.0)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = env::var("LLAMA_URL") {
            config.llama_url = url;
        }

        if let Ok(servers) = env::var("MCP_SERVERS") {
            config.mcp_servers = parse_mcp_servers(&servers);
        }

        if let Ok(max_iter) = env::var("MAX_TOOL_ITERATIONS") {
            if let Ok(n) = max_iter.parse() {
                config.max_tool_iterations = n;
            }
        }

        if let Ok(timeout) = env::var("TIMEOUT_SECS") {
            if let Ok(secs) = timeout.parse() {
                config.timeout = Duration::from_secs(secs);
            }
        }

        if let Ok(ttl) = env::var("RESPONSE_TTL_SECS") {
            if let Ok(secs) = ttl.parse() {
                config.response_ttl = Duration::from_secs(secs);
            }
        }

        if let Ok(port) = env::var("PORT") {
            if let Ok(p) = port.parse() {
                config.port = p;
            }
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

    /// Set the llama.cpp URL.
    pub fn llama_url(mut self, url: &str) -> Self {
        self.llama_url = url.to_string();
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

    /// Create default configuration with standard MCP servers.
    ///
    /// Includes:
    /// - mcp-weather on port 9100
    /// - mcp-web-search on port 9110
    /// - mcp-code-interpreter on port 9120
    /// - mcp-reader on port 9130
    pub fn with_standard_mcp_servers() -> Self {
        Self::default()
            .add_mcp_server("weather", "http://mcp-weather:8000/mcp")
            .add_mcp_server("web_search", "http://mcp-web-search:8000/mcp")
            .add_mcp_server("code_interpreter", "http://mcp-code-interpreter:8000/mcp")
            .add_mcp_server("reader", "http://mcp-reader:8000/mcp")
    }
}

/// Parse MCP_SERVERS environment variable.
///
/// Format: "name:url" or "name:url:prefix", comma-separated
/// Example: "weather:http://mcp-weather:8000/mcp,search:http://mcp-search:8000/mcp:search_"
fn parse_mcp_servers(env_value: &str) -> Vec<McpServerConfig> {
    env_value
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                return None;
            }

            let parts: Vec<&str> = s.splitn(3, ':').collect();
            match parts.len() {
                // "name:url" - but URL has colons, so we need smarter parsing
                _ if s.contains("://") => {
                    // Find the name (before first colon before ://)
                    if let Some(protocol_pos) = s.find("://") {
                        // Find the colon that comes before the protocol
                        if let Some(name_end) = s[..protocol_pos].rfind(':') {
                            let name = &s[..name_end];
                            let rest = &s[name_end + 1..];

                            // Check if there's a prefix after the URL
                            // URL ends at next space or colon after the port/path
                            if let Some(last_colon) = rest.rfind(':') {
                                // Check if this colon is part of URL or is prefix separator
                                let after_last_colon = &rest[last_colon + 1..];
                                // If it doesn't contain / and isn't numeric, it's a prefix
                                if !after_last_colon.contains('/')
                                    && after_last_colon.parse::<u16>().is_err()
                                    && !after_last_colon.is_empty()
                                {
                                    let url = &rest[..last_colon];
                                    let prefix = after_last_colon;
                                    return Some(McpServerConfig::new(name, url).with_prefix(prefix));
                                }
                            }
                            return Some(McpServerConfig::new(name, rest));
                        }
                    }
                    None
                }
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let config = Config::default();
        assert!(config.llama_url.contains("llama-server"));
        assert_eq!(config.port, 8000);
        assert_eq!(config.max_tool_iterations, 10);
    }

    #[test]
    fn builder_pattern_works() {
        let config = Config::default()
            .llama_url("http://custom:9000")
            .port(3000)
            .max_tool_iterations(5)
            .add_mcp_server("test", "http://test:8000/mcp");

        assert_eq!(config.llama_url, "http://custom:9000");
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_tool_iterations, 5);
        assert_eq!(config.mcp_servers.len(), 1);
    }

    #[test]
    fn parse_mcp_servers_simple() {
        let servers = parse_mcp_servers("weather:http://mcp-weather:8000/mcp");
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "weather");
        assert_eq!(servers[0].url, "http://mcp-weather:8000/mcp");
        assert!(servers[0].tool_prefix.is_none());
    }

    #[test]
    fn parse_mcp_servers_with_prefix() {
        let servers = parse_mcp_servers("weather:http://mcp-weather:8000/mcp:weather_");
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "weather");
        assert_eq!(servers[0].url, "http://mcp-weather:8000/mcp");
        assert_eq!(servers[0].tool_prefix, Some("weather_".to_string()));
    }

    #[test]
    fn parse_mcp_servers_multiple() {
        let servers = parse_mcp_servers(
            "weather:http://mcp-weather:8000/mcp,search:http://mcp-search:8000/mcp:search_",
        );
        assert_eq!(servers.len(), 2);
        assert_eq!(servers[0].name, "weather");
        assert_eq!(servers[1].name, "search");
        assert_eq!(servers[1].tool_prefix, Some("search_".to_string()));
    }

    #[test]
    fn standard_mcp_servers_includes_expected() {
        let config = Config::with_standard_mcp_servers();
        assert_eq!(config.mcp_servers.len(), 4);

        let names: Vec<_> = config.mcp_servers.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"weather"));
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"code_interpreter"));
        assert!(names.contains(&"reader"));
    }
}
