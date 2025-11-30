//! OpenAI Responses API adapter for llama.cpp.
//!
//! This crate provides a translation layer that converts OpenAI Responses API
//! requests into Chat Completions format for llama.cpp, and converts the
//! responses back. It includes server-side tool calling execution via MCP.

pub mod config;
pub mod execution;
pub mod mcp;
pub mod models;
pub mod server;
pub mod state;
pub mod translation;
