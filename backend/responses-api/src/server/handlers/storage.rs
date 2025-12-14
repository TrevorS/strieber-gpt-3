//! HTTP handlers for generic storage endpoints.
//!
//! Provides a simple key-value store API for MCP servers to persist data.
//!
//! # URL Structure
//!
//! ```text
//! POST   /v1/storage/jobs                  # Create record
//! GET    /v1/storage/jobs                  # List records
//! GET    /v1/storage/jobs/{id}             # Get single record
//! DELETE /v1/storage/jobs/{id}             # Delete record
//! ```
//!
//! Collections are just simple names: `jobs`, `datasets`, `tasks`, etc.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::server::handlers::AppState;
use crate::storage::generic::StoredRecord;

/// Request body for saving data.
#[derive(Debug, Deserialize)]
pub struct SaveRequest {
    pub id: String,
    pub data: serde_json::Value,
}

/// Query parameters for listing records.
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub status: Option<String>, // Optional status filter (stored in data.status)
}

fn default_limit() -> usize {
    100
}

/// Response for save operation.
#[derive(Debug, Serialize)]
pub struct SaveResponse {
    pub id: String,
    pub collection: String,
    pub created: bool,
}

/// Response for list operation.
#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub collection: String,
    pub records: Vec<StoredRecord>,
    pub count: usize,
}

/// Response for delete operation.
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub id: String,
    pub collection: String,
    pub deleted: bool,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

/// POST /v1/storage/{collection} - Create or update a record.
pub async fn save_record(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<SaveRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Check if record exists to determine if this is create or update
    let exists = state.generic_store.get(&collection, &req.id).is_some();

    // Save the record
    state.generic_store.save(&collection, &req.id, &req.data);

    tracing::debug!(
        collection = %collection,
        id = %req.id,
        created = !exists,
        "Saved record"
    );

    Ok((
        StatusCode::OK,
        Json(SaveResponse {
            id: req.id,
            collection,
            created: !exists,
        }),
    ))
}

/// GET /v1/storage/{collection}/{id} - Get a single record.
pub async fn get_record(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let record = state.generic_store.get(&collection, &id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Record {}/{} not found", collection, id)
                }
            })),
        )
    })?;

    Ok((StatusCode::OK, Json(record)))
}

/// GET /v1/storage/{collection} - List records in a collection.
pub async fn list_records(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let mut records = state.generic_store.list(&collection, query.limit);

    // Apply optional status filter
    if let Some(ref status_filter) = query.status {
        records.retain(|r| {
            r.data
                .get("status")
                .and_then(|s| s.as_str())
                .map(|s| s == status_filter)
                .unwrap_or(false)
        });
    }

    let count = records.len();

    tracing::debug!(
        collection = %collection,
        count = count,
        limit = query.limit,
        status_filter = ?query.status,
        "Listed records"
    );

    Ok((
        StatusCode::OK,
        Json(ListResponse {
            collection,
            records,
            count,
        }),
    ))
}

/// DELETE /v1/storage/{collection}/{id} - Delete a record.
pub async fn delete_record(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let deleted = state.generic_store.delete(&collection, &id);

    if deleted {
        tracing::debug!(
            collection = %collection,
            id = %id,
            "Deleted record"
        );

        Ok((
            StatusCode::OK,
            Json(DeleteResponse {
                id,
                collection,
                deleted: true,
            }),
        ))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Record {}/{} not found", collection, id)
                }
            })),
        ))
    }
}
