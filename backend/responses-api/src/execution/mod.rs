//! Tool execution loop for Responses API.
//!
//! Orchestrates the full request lifecycle:
//! 1. Translate Responses API request → Chat Completions
//! 2. Call llama.cpp
//! 3. If tool calls returned, execute via MCP
//! 4. Loop until completion or max iterations
//! 5. Translate Chat Completions response → Responses API

mod executor;

pub use executor::{Executor, ExecutorConfig, ExecutionError};
