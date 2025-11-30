//! Request execution for Responses API.
//!
//! Provides both synchronous and streaming execution modes.

mod executor;
mod streaming;

pub use executor::{ExecutionError, Executor, ExecutorConfig};
pub use streaming::execute_streaming;
