//! HTTP handlers for container file endpoints.

use std::sync::Arc;

use axum::{
    Json,
    body::Body,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::server::AppState;

/// GET /v1/containers/{container_id}/files/{file_id}/content
/// Download a file from a container.
pub async fn download_file(
    State(state): State<Arc<AppState>>,
    Path((container_id, file_id)): Path<(String, String)>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Check if container exists
    if !state.containers.exists(&container_id) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Container {} not found", container_id)
                }
            })),
        ));
    }

    // Get file content
    let (content, mime_type) = state
        .containers
        .get_file_content(&container_id, &file_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "type": "not_found",
                        "message": format!("File {} not found in container {}", file_id, container_id)
                    }
                })),
            )
        })?;

    // Get filename for Content-Disposition header
    let filename = state
        .containers
        .get_file_metadata(&container_id, &file_id)
        .map(|(name, _, _)| name)
        .unwrap_or_else(|| file_id.clone());

    // Build response with appropriate headers
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, mime_type)
        .header(
            header::CONTENT_DISPOSITION,
            format!("inline; filename=\"{}\"", filename),
        )
        .header(header::CONTENT_LENGTH, content.len())
        .body(Body::from(content))
        .unwrap();

    Ok(response)
}

/// GET /v1/containers/{container_id}/files
/// List files in a container.
pub async fn list_container_files(
    State(state): State<Arc<AppState>>,
    Path(container_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Check if container exists
    if !state.containers.exists(&container_id) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "type": "not_found",
                    "message": format!("Container {} not found", container_id)
                }
            })),
        ));
    }

    // List files
    let file_ids = state
        .containers
        .list_files(&container_id)
        .unwrap_or_default();

    let files: Vec<_> = file_ids
        .iter()
        .filter_map(|id| {
            state.containers.get_file_metadata(&container_id, id).map(
                |(filename, mime_type, size)| {
                    json!({
                        "id": id,
                        "object": "container.file",
                        "container_id": container_id,
                        "filename": filename,
                        "mime_type": mime_type,
                        "size": size
                    })
                },
            )
        })
        .collect();

    Ok((
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": files
        })),
    ))
}
