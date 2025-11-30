//! MCP (Model Context Protocol) client for tool execution.
//!
//! Uses the official rmcp crate with Streamable HTTP transport to call
//! MCP servers for tool execution.

mod client;

pub use client::{McpClient, McpError, McpServerConfig};
pub use rmcp::model::{CallToolResult, Tool as McpTool};
