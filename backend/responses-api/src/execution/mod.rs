//! Request execution for Responses API.
//!
//! Provides both synchronous and streaming execution modes.

mod context;
mod executor;
mod streaming;
pub mod title_generator;

pub use context::{ChainErrorKind, ChainResolutionError, DEFAULT_MAX_CHAIN_DEPTH, resolve_chain};
pub use executor::{ExecutionError, Executor, ExecutorConfig, GeneratedFile};
pub use streaming::execute_streaming;
