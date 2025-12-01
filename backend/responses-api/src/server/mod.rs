//! HTTP API server for Responses API.
//!
//! Implements:
//! - POST /v1/responses - Create a response
//! - GET /v1/responses/{id} - Get a response by ID
//! - DELETE /v1/responses/{id} - Delete a response

pub mod handlers;
mod routes;

pub use handlers::AppState;
pub use routes::create_router;
