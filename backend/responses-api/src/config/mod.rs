//! Configuration for the Responses API service.
//!
//! Supports configuration via:
//! - Environment variables (MCP_CONFIG, MODELS_CONFIG, etc.)
//! - Programmatic configuration

use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

use crate::mcp::McpServerConfig;

/// Configuration for a model backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "gpt-4o", "gpt-oss-120b")
    pub id: String,
    /// Backend URL for this model (e.g., "http://llama-server:8000")
    pub url: String,
    /// Optional API key for authentication (Bearer token)
    #[serde(default)]
    pub api_key: Option<String>,
    /// Owner/provider of this model (default: "local")
    #[serde(default = "default_owned_by")]
    pub owned_by: String,
    /// Whether this model supports vision/image inputs
    #[serde(default)]
    pub supports_vision: bool,
    /// Which tools this model supports.
    /// None = all tools, Some([]) = no tools, Some(["web_search"]) = specific tools
    #[serde(default)]
    pub supported_tools: Option<Vec<String>>,
}

fn default_owned_by() -> String {
    "local".to_string()
}

impl ModelConfig {
    /// Create a new model configuration.
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            api_key: None,
            owned_by: "local".to_string(),
            supports_vision: false,
            supported_tools: None,
        }
    }

    /// Set the API key for this model.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the owner for this model.
    pub fn with_owned_by(mut self, owned_by: impl Into<String>) -> Self {
        self.owned_by = owned_by.into();
        self
    }

    /// Enable vision/image input support for this model.
    pub fn with_vision(mut self) -> Self {
        self.supports_vision = true;
        self
    }
}

/// Main configuration for the Responses API service.
#[derive(Debug, Clone)]
pub struct Config {
    /// Available models and their backend configurations
    pub models: Vec<ModelConfig>,
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
            models: Vec::new(),
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

/// JSON structure for MODELS_CONFIG environment variable.
#[derive(Debug, Deserialize)]
struct ModelsConfigJson {
    models: Vec<ModelConfig>,
}

impl Config {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `MODELS_CONFIG`: JSON object with model configurations (required)
    ///   Example: {"models":[{"id":"gpt-oss-120b","url":"http://llama-server:8000"}]}
    /// - `MCP_CONFIG`: JSON object with MCP server configuration
    ///   Example: {"servers":[{"name":"weather","url":"http://mcp-weather:8000/mcp"}]}
    /// - `MAX_TOOL_ITERATIONS`: Maximum tool calling iterations (default: 10)
    /// - `TIMEOUT_SECS`: HTTP request timeout in seconds (default: 300)
    /// - `RESPONSE_TTL_SECS`: Response TTL in seconds (default: 3600)
    /// - `PORT`: Port to listen on (default: 8000)
    /// - `HOST`: Host to bind to (default: 0.0.0.0)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Parse models configuration from JSON env var
        if let Ok(json) = env::var("MODELS_CONFIG") {
            match serde_json::from_str::<ModelsConfigJson>(&json) {
                Ok(models_config) => config.models = models_config.models,
                Err(e) => tracing::error!("Failed to parse MODELS_CONFIG: {}", e),
            }
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

    /// Get a model configuration by ID.
    pub fn get_model(&self, model_id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.id == model_id)
    }

    /// Add a model to the configuration.
    pub fn add_model(mut self, model: ModelConfig) -> Self {
        self.models.push(model);
        self
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
        assert!(config.models.is_empty());
        assert_eq!(config.port, 8000);
        assert_eq!(config.max_tool_iterations, 10);
    }

    #[test]
    fn builder_pattern_works() {
        let config = Config::default()
            .add_model(ModelConfig::new("test-model", "http://localhost:8000"))
            .port(3000)
            .max_tool_iterations(5)
            .add_mcp_server("test", "http://test:8000/mcp");

        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "test-model");
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_tool_iterations, 5);
        assert_eq!(config.mcp_servers.len(), 1);
    }

    #[test]
    fn get_model_by_id() {
        let config = Config::default()
            .add_model(ModelConfig::new("model-a", "http://a:8000"))
            .add_model(ModelConfig::new("model-b", "http://b:8000"));

        assert!(config.get_model("model-a").is_some());
        assert_eq!(config.get_model("model-a").unwrap().url, "http://a:8000");
        assert!(config.get_model("model-b").is_some());
        assert!(config.get_model("model-c").is_none());
    }

    #[test]
    fn model_config_builder() {
        let model = ModelConfig::new("gpt-4o", "https://api.openai.com")
            .with_api_key("sk-test")
            .with_owned_by("openai");

        assert_eq!(model.id, "gpt-4o");
        assert_eq!(model.url, "https://api.openai.com");
        assert_eq!(model.api_key, Some("sk-test".to_string()));
        assert_eq!(model.owned_by, "openai");
    }

    #[test]
    fn parse_models_config_json() {
        let json = r#"{"models":[{"id":"gpt-oss-120b","url":"http://llama-server:8000"}]}"#;
        let config: ModelsConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "gpt-oss-120b");
        assert_eq!(config.models[0].url, "http://llama-server:8000");
        assert_eq!(config.models[0].owned_by, "local"); // default
    }

    #[test]
    fn parse_models_config_with_api_key() {
        let json = r#"{"models":[{"id":"gpt-4o","url":"https://api.openai.com","api_key":"sk-test","owned_by":"openai"}]}"#;
        let config: ModelsConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].id, "gpt-4o");
        assert_eq!(config.models[0].api_key, Some("sk-test".to_string()));
        assert_eq!(config.models[0].owned_by, "openai");
    }

    #[test]
    fn parse_models_config_multiple() {
        let json = r#"{"models":[{"id":"model-a","url":"http://a:8000"},{"id":"model-b","url":"http://b:8000"}]}"#;
        let config: ModelsConfigJson = serde_json::from_str(json).unwrap();
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].id, "model-a");
        assert_eq!(config.models[1].id, "model-b");
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
        assert_eq!(config.servers[0].tool_prefix, Some("search_".to_string()));
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
