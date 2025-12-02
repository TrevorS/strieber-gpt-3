//! Request execution for Responses API.
//!
//! Provides both synchronous and streaming execution modes.

mod context;
mod executor;
mod streaming;

pub use context::{resolve_chain, ChainErrorKind, ChainResolutionError, DEFAULT_MAX_CHAIN_DEPTH};
pub use executor::{ExecutionError, Executor, ExecutorConfig, GeneratedFile};
pub use streaming::execute_streaming;
