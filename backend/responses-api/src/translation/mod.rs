//! Translation between Responses API and Chat Completions formats.
//!
//! This module handles bidirectional conversion:
//! - Responses API requests → Chat Completions requests
//! - Chat Completions responses → Responses API responses (for clients)

mod ids;
mod request;
mod response;

pub use ids::*;
pub use request::*;
pub use response::*;
