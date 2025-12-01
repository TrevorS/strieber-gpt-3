//! OpenAI Responses API adapter.
//!
//! This crate provides a translation layer that converts OpenAI Responses API
//! requests into Chat Completions format for any compatible inference backend,
//! and converts the responses back. It includes server-side tool calling via MCP.

pub mod config;
pub mod containers;
pub mod execution;
pub mod mcp;
pub mod models;
pub mod server;
pub mod state;
pub mod translation;
