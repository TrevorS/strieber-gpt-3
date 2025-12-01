//! Container file storage for code interpreter outputs.
//!
//! Containers are file storage buckets that hold generated files (images, etc.)
//! from code interpreter executions. They don't maintain Python state - just files.

mod handlers;
mod store;

pub use handlers::*;
pub use store::*;
