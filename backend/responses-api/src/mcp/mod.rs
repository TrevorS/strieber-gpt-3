//! MCP (Model Context Protocol) client for tool execution.
//!
//! Uses the official rmcp crate with Streamable HTTP transport to call
//! MCP servers for tool execution.

mod client;

pub use client::{McpClient, McpServerConfig, McpError};
pub use rmcp::model::{Tool as McpTool, CallToolResult};
