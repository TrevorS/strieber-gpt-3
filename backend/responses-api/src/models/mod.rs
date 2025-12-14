//! OpenAI Responses API type definitions.
//!
//! This module contains all the serde-compatible types needed to implement
//! the OpenAI Responses API, translating between the Responses format and
//! the Chat Completions format used by any compatible inference engine.

mod chat;
mod common;
mod conversation;
mod events;
pub mod item_expansion;
mod request;
mod response;
mod tools;

pub use chat::*;
pub use common::*;
pub use conversation::*;
pub use events::*;
pub use request::*;
pub use response::*;
pub use tools::*;
