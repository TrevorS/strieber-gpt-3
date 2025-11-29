//! OpenAI Responses API type definitions.
//!
//! This module contains all the serde-compatible types needed to implement
//! the OpenAI Responses API, translating between the Responses format and
//! the Chat Completions format used by llama.cpp.

mod chat;
mod events;
mod request;
mod response;
mod tools;

pub use chat::*;
pub use events::*;
pub use request::*;
pub use response::*;
pub use tools::*;
