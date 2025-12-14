//! Item expansion logic for the `include` query parameter.
//!
//! When clients request additional data via the `include` parameter,
//! this module handles expanding nested content in conversation items.
//!
//! Supported include values:
//! - `message.input_image.image_url` - Include full image URLs for input images
//! - `code_interpreter_call.outputs` - Include code interpreter outputs (images, logs)
//! - `web_search_call.action.sources` - Include web search sources
//! - `file_search_call.results` - Include file search results
//! - `reasoning.encrypted_content` - Include encrypted reasoning content
//! - `computer_call_output.output.image_url` - Include computer call screenshots

use serde_json::{Value, json};

use crate::containers::ContainerStore;

use super::common::ItemsListQuery;
use super::conversation::{
    ConversationItem, ConversationItemContent, CreateItemsQuery, GetItemQuery,
};
use super::request::{ContentPart, InputItem, MessageContent};

/// Expansion options derived from the `include` parameter.
#[derive(Debug, Clone, Default)]
pub struct ExpansionOptions {
    /// Include full image URLs for input images
    pub input_image_url: bool,
    /// Include code interpreter outputs
    pub code_interpreter_outputs: bool,
    /// Include web search sources
    pub web_search_sources: bool,
    /// Include file search results
    pub file_search_results: bool,
    /// Include encrypted reasoning content
    pub reasoning_encrypted: bool,
    /// Include computer call output image URLs
    pub computer_call_image_url: bool,
}

impl From<&ItemsListQuery> for ExpansionOptions {
    fn from(query: &ItemsListQuery) -> Self {
        Self {
            input_image_url: query.includes("message.input_image.image_url"),
            code_interpreter_outputs: query.includes("code_interpreter_call.outputs"),
            web_search_sources: query.includes("web_search_call.action.sources"),
            file_search_results: query.includes("file_search_call.results"),
            reasoning_encrypted: query.includes("reasoning.encrypted_content"),
            computer_call_image_url: query.includes("computer_call_output.output.image_url"),
        }
    }
}

impl ExpansionOptions {
    /// Create options from a CreateItemsQuery.
    pub fn from_create_items_query(query: &CreateItemsQuery) -> Self {
        Self {
            input_image_url: query.includes("message.input_image.image_url"),
            code_interpreter_outputs: query.includes("code_interpreter_call.outputs"),
            web_search_sources: query.includes("web_search_call.action.sources"),
            file_search_results: query.includes("file_search_call.results"),
            reasoning_encrypted: query.includes("reasoning.encrypted_content"),
            computer_call_image_url: query.includes("computer_call_output.output.image_url"),
        }
    }

    /// Create options from a GetItemQuery.
    pub fn from_get_item_query(query: &GetItemQuery) -> Self {
        Self {
            input_image_url: query.includes("message.input_image.image_url"),
            code_interpreter_outputs: query.includes("code_interpreter_call.outputs"),
            web_search_sources: query.includes("web_search_call.action.sources"),
            file_search_results: query.includes("file_search_call.results"),
            reasoning_encrypted: query.includes("reasoning.encrypted_content"),
            computer_call_image_url: query.includes("computer_call_output.output.image_url"),
        }
    }

    /// Check if any expansion is requested.
    pub fn has_any(&self) -> bool {
        self.input_image_url
            || self.code_interpreter_outputs
            || self.web_search_sources
            || self.file_search_results
            || self.reasoning_encrypted
            || self.computer_call_image_url
    }
}

/// Expand a conversation item based on the include options.
///
/// This function modifies the item in-place to include or exclude
/// nested content based on what was requested via the `include` parameter.
pub fn expand_item(
    item: &mut ConversationItem,
    options: &ExpansionOptions,
    containers: &ContainerStore,
    base_url: &str,
) {
    match &mut item.content {
        ConversationItemContent::Input(input) => {
            expand_input_item(input, options, containers, base_url);
        }
        ConversationItemContent::Output(output) => {
            expand_output_item(output, options, containers, base_url);
        }
    }
}

/// Expand items in a list.
pub fn expand_items(
    items: &mut [ConversationItem],
    options: &ExpansionOptions,
    containers: &ContainerStore,
    base_url: &str,
) {
    if !options.has_any() {
        return;
    }

    for item in items.iter_mut() {
        expand_item(item, options, containers, base_url);
    }
}

/// Expand input item content.
fn expand_input_item(
    item: &mut InputItem,
    options: &ExpansionOptions,
    containers: &ContainerStore,
    base_url: &str,
) {
    if let InputItem::Message(msg) = item
        && options.input_image_url
    {
        expand_message_images(&mut msg.content, containers, base_url);
    }
    // Other input types don't have expandable content currently
}

/// Expand image URLs in message content.
fn expand_message_images(
    content: &mut MessageContent,
    containers: &ContainerStore,
    base_url: &str,
) {
    if let MessageContent::Parts(parts) = content {
        for part in parts.iter_mut() {
            if let ContentPart::InputImage { image_url } = part {
                // If the URL looks like a file reference, expand it
                if let Some(expanded) = expand_file_url(&image_url.url, containers, base_url) {
                    image_url.url = expanded;
                }
            }
        }
    }
}

/// Expand output item content (stored as JSON).
fn expand_output_item(
    output: &mut Value,
    options: &ExpansionOptions,
    containers: &ContainerStore,
    base_url: &str,
) {
    let item_type = output.get("type").and_then(|t| t.as_str());

    match item_type {
        Some("code_interpreter_call") => {
            if options.code_interpreter_outputs {
                expand_code_interpreter_outputs(output, containers, base_url);
            } else {
                // Strip outputs if not requested
                strip_field(output, "outputs");
            }
        }
        Some("web_search_call") => {
            if !options.web_search_sources {
                // Strip sources from action if not requested
                if let Some(action) = output.get_mut("action") {
                    strip_field(action, "sources");
                }
            }
        }
        Some("file_search_call") => {
            if !options.file_search_results {
                strip_field(output, "results");
            }
        }
        Some("reasoning") => {
            if !options.reasoning_encrypted {
                strip_field(output, "encrypted_content");
            }
        }
        Some("computer_call_output") => {
            if options.computer_call_image_url {
                expand_computer_call_output(output, containers, base_url);
            }
        }
        _ => {}
    }
}

/// Expand code interpreter outputs with full URLs.
fn expand_code_interpreter_outputs(
    output: &mut Value,
    containers: &ContainerStore,
    base_url: &str,
) {
    if let Some(outputs) = output.get_mut("outputs").and_then(|o| o.as_array_mut()) {
        for out in outputs.iter_mut() {
            if out.get("type").and_then(|t| t.as_str()) == Some("image")
                && let Some(file_id) = out.get("file_id").and_then(|f| f.as_str())
                && let Some(url) = expand_container_file_url(file_id, containers, base_url)
            {
                out["url"] = json!(url);
            }
        }
    }
}

/// Expand computer call output with image URL.
fn expand_computer_call_output(output: &mut Value, containers: &ContainerStore, base_url: &str) {
    if let Some(out) = output.get_mut("output") {
        if let Some(file_id) = out.get("file_id").and_then(|f| f.as_str()) {
            if let Some(url) = expand_container_file_url(file_id, containers, base_url) {
                out["image_url"] = json!(url);
            }
        } else if let Some(image_url) = out.get("image_url").and_then(|u| u.as_str()) {
            // If it's a file reference, expand it
            if let Some(url) = expand_file_url(image_url, containers, base_url) {
                out["image_url"] = json!(url);
            }
        }
    }
}

/// Expand a file URL if it's a container file reference.
fn expand_file_url(url: &str, containers: &ContainerStore, base_url: &str) -> Option<String> {
    // Check if it's a container file reference (format: container_id/file_id)
    if (url.starts_with("cntr_") || url.starts_with("cfile_"))
        && let Some((container_id, file_id)) = url.split_once('/')
        && containers.exists(container_id)
    {
        return Some(format!(
            "{}/v1/containers/{}/files/{}/content",
            base_url, container_id, file_id
        ));
    }
    None
}

/// Expand a container file ID to a full URL.
fn expand_container_file_url(
    file_id: &str,
    containers: &ContainerStore,
    base_url: &str,
) -> Option<String> {
    // For standalone file_id, we need to find which container it's in
    // This requires a lookup across all containers (expensive but necessary)
    // For now, check if the file_id contains container info

    // If file_id contains container info in format "container_id:file_id"
    if let Some((container_id, actual_file_id)) = file_id.split_once(':')
        && containers.exists(container_id)
    {
        return Some(format!(
            "{}/v1/containers/{}/files/{}/content",
            base_url, container_id, actual_file_id
        ));
    }

    // Otherwise, we can't expand without container context
    // The caller should have stored the container_id alongside the file_id
    None
}

/// Strip a field from a JSON object.
fn strip_field(obj: &mut Value, field: &str) {
    if let Some(map) = obj.as_object_mut() {
        map.remove(field);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expansion_options_from_query() {
        let query = ItemsListQuery {
            after: None,
            limit: 20,
            order: crate::models::SortOrder::Desc,
            include: Some(vec![
                "code_interpreter_call.outputs".to_string(),
                "reasoning.encrypted_content".to_string(),
            ]),
        };

        let options = ExpansionOptions::from(&query);
        assert!(options.code_interpreter_outputs);
        assert!(options.reasoning_encrypted);
        assert!(!options.input_image_url);
        assert!(!options.web_search_sources);
    }

    #[test]
    fn expansion_options_has_any() {
        let empty = ExpansionOptions::default();
        assert!(!empty.has_any());

        let with_one = ExpansionOptions {
            code_interpreter_outputs: true,
            ..Default::default()
        };
        assert!(with_one.has_any());
    }

    #[test]
    fn strip_field_removes_from_object() {
        let mut obj = json!({
            "type": "reasoning",
            "content": [],
            "encrypted_content": "secret"
        });

        strip_field(&mut obj, "encrypted_content");

        assert!(obj.get("encrypted_content").is_none());
        assert!(obj.get("type").is_some());
        assert!(obj.get("content").is_some());
    }

    #[test]
    fn expand_output_strips_reasoning_encrypted_when_not_requested() {
        let containers = ContainerStore::new();
        let options = ExpansionOptions::default(); // reasoning_encrypted is false

        let mut output = json!({
            "type": "reasoning",
            "id": "rs_123",
            "content": [{"type": "reasoning_text", "text": "thinking..."}],
            "encrypted_content": "encrypted_data_here"
        });

        expand_output_item(&mut output, &options, &containers, "http://localhost");

        assert!(output.get("encrypted_content").is_none());
    }

    #[test]
    fn expand_output_keeps_reasoning_encrypted_when_requested() {
        let containers = ContainerStore::new();
        let options = ExpansionOptions {
            reasoning_encrypted: true,
            ..Default::default()
        };

        let mut output = json!({
            "type": "reasoning",
            "id": "rs_123",
            "content": [],
            "encrypted_content": "encrypted_data_here"
        });

        expand_output_item(&mut output, &options, &containers, "http://localhost");

        assert_eq!(
            output.get("encrypted_content").and_then(|v| v.as_str()),
            Some("encrypted_data_here")
        );
    }

    #[test]
    fn expand_output_strips_web_search_sources_when_not_requested() {
        let containers = ContainerStore::new();
        let options = ExpansionOptions::default();

        let mut output = json!({
            "type": "web_search_call",
            "id": "ws_123",
            "status": "completed",
            "action": {
                "query": "rust programming",
                "sources": [
                    {"url": "https://example.com", "title": "Example"}
                ]
            }
        });

        expand_output_item(&mut output, &options, &containers, "http://localhost");

        let action = output.get("action").unwrap();
        assert!(action.get("sources").is_none());
        assert!(action.get("query").is_some());
    }

    #[test]
    fn expand_output_keeps_web_search_sources_when_requested() {
        let containers = ContainerStore::new();
        let options = ExpansionOptions {
            web_search_sources: true,
            ..Default::default()
        };

        let mut output = json!({
            "type": "web_search_call",
            "id": "ws_123",
            "status": "completed",
            "action": {
                "query": "rust programming",
                "sources": [
                    {"url": "https://example.com", "title": "Example"}
                ]
            }
        });

        expand_output_item(&mut output, &options, &containers, "http://localhost");

        let action = output.get("action").unwrap();
        assert!(action.get("sources").is_some());
    }

    #[test]
    fn expand_code_interpreter_adds_url_for_container_files() {
        let containers = ContainerStore::new();
        let container_id = containers.create();

        // Add a test file
        let file_id = containers
            .add_file(
                &container_id,
                "output.png".to_string(),
                vec![0x89, 0x50, 0x4E, 0x47], // PNG magic bytes
                "image/png",
            )
            .unwrap();

        let options = ExpansionOptions {
            code_interpreter_outputs: true,
            ..Default::default()
        };

        // Use container_id:file_id format for lookup
        let mut output = json!({
            "type": "code_interpreter_call",
            "id": "ci_123",
            "status": "completed",
            "code": "print('hello')",
            "outputs": [
                {"type": "image", "file_id": format!("{}:{}", container_id, file_id)}
            ]
        });

        expand_output_item(&mut output, &options, &containers, "http://localhost:8000");

        let outputs = output.get("outputs").unwrap().as_array().unwrap();
        let image = &outputs[0];

        let url = image.get("url").and_then(|u| u.as_str());
        assert!(url.is_some());
        assert!(url.unwrap().contains("/v1/containers/"));
        assert!(url.unwrap().contains("/files/"));
        assert!(url.unwrap().contains("/content"));
    }
}
