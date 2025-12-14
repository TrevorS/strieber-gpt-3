//! Automatic conversation title generation using a lightweight task model.

use crate::config::ModelConfig;
use crate::models::{
    ContentPart, Input, InputItem, MessageContent, OutputContent, OutputItem, Role,
};
use crate::state::{ConversationStore, InMemoryConversationStore};
use reqwest::Client;

const TITLE_PROMPT: &str = "Generate a brief title (3-6 words) for this conversation. Reply with ONLY the title, nothing else.";

/// Generate a conversation title using the task model.
///
/// Makes a non-streaming chat completion request to the task model
/// with the first user message and assistant response.
pub async fn generate_title(
    task_model: &ModelConfig,
    http_client: &Client,
    user_message: &str,
    assistant_response: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Build chat completion request
    let prompt = format!(
        "{}\n\nUser: {}\nAssistant: {}",
        TITLE_PROMPT,
        truncate(user_message, 200),
        truncate(assistant_response, 300)
    );

    let request_body = serde_json::json!({
        "model": task_model.id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.7,
    });

    tracing::info!(
        model = %task_model.id,
        url = %task_model.url,
        "Generating conversation title"
    );

    // POST to task model's chat/completions endpoint
    let response = http_client
        .post(format!("{}/v1/chat/completions", task_model.url))
        .json(&request_body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Title generation failed: {} - {}", status, body).into());
    }

    let json: serde_json::Value = response.json().await?;
    let title = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("New Chat")
        .trim()
        .to_string();

    tracing::info!(title = %title, "Generated conversation title");

    Ok(title)
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a safe UTF-8 boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// Check if title generation should run for this conversation.
///
/// Returns true if:
/// - The conversation exists
/// - The title is still "New Chat" or empty
/// - This is the first exchange (only 2 items: user message + assistant response)
pub fn should_generate_title(conv_store: &InMemoryConversationStore, conv_id: &str) -> bool {
    conv_store
        .get(conv_id)
        .map(|c| {
            let title = c.metadata.as_ref().and_then(|m| m.inner().get("title"));
            let is_default = title.is_none() || title == Some(&"New Chat".to_string());

            // Count items to check if this is the first exchange
            use crate::models::{PaginationQuery, SortOrder};
            let pagination = PaginationQuery {
                after: None,
                limit: 10,
                order: SortOrder::Asc,
            };
            let item_count = conv_store
                .list_items(conv_id, &pagination)
                .map(|l| l.data.len())
                .unwrap_or(0);

            // Only generate on first exchange (2 items = user msg + assistant response)
            is_default && item_count <= 2
        })
        .unwrap_or(false)
}

/// Find the first model with "task" capability from the config.
pub fn find_task_model(models: &[ModelConfig]) -> Option<&ModelConfig> {
    models
        .iter()
        .find(|m| m.capabilities.contains(&"task".to_string()))
}

/// Extract the first user message text from the request Input.
pub fn extract_first_user_message(input: &Input) -> String {
    match input {
        Input::Text(text) => text.clone(),
        Input::Items(items) => {
            for item in items {
                if let InputItem::Message(msg) = item
                    && msg.role == Role::User
                {
                    return extract_text_from_message_content(&msg.content);
                }
            }
            String::new()
        }
        Input::Empty => String::new(),
    }
}

/// Extract text content from MessageContent.
fn extract_text_from_message_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => {
            for part in parts {
                if let ContentPart::InputText { text } = part {
                    return text.clone();
                }
            }
            String::new()
        }
    }
}

/// Extract the assistant response text from output items.
pub fn extract_assistant_response(output: &[OutputItem]) -> String {
    for item in output {
        if let OutputItem::Message(msg) = item {
            for content in &msg.content {
                if let OutputContent::OutputText { text, .. } = content {
                    return text.clone();
                }
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_ascii() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_utf8() {
        // "héllo" has a 2-byte character
        let s = "héllo";
        // Truncating in the middle of the é should back up to safe boundary
        let truncated = truncate(s, 2);
        assert!(truncated.len() <= 2);
    }

    #[test]
    fn test_find_task_model() {
        let models = vec![
            ModelConfig {
                id: "reasoning-model".to_string(),
                url: "http://a:8000".to_string(),
                api_key: None,
                owned_by: "local".to_string(),
                supports_vision: false,
                supported_tools: None,
                reasoning: None,
                capabilities: vec!["reasoning".to_string()],
            },
            ModelConfig {
                id: "task-model".to_string(),
                url: "http://b:8000".to_string(),
                api_key: None,
                owned_by: "local".to_string(),
                supports_vision: true,
                supported_tools: None,
                reasoning: None,
                capabilities: vec!["task".to_string(), "vision".to_string()],
            },
        ];

        let task = find_task_model(&models);
        assert!(task.is_some());
        assert_eq!(task.unwrap().id, "task-model");
    }

    #[test]
    fn test_find_task_model_none() {
        let models = vec![ModelConfig {
            id: "reasoning-model".to_string(),
            url: "http://a:8000".to_string(),
            api_key: None,
            owned_by: "local".to_string(),
            supports_vision: false,
            supported_tools: None,
            reasoning: None,
            capabilities: vec!["reasoning".to_string()],
        }];

        let task = find_task_model(&models);
        assert!(task.is_none());
    }

    #[test]
    fn test_extract_first_user_message_text() {
        let input = Input::Text("Hello world".to_string());
        assert_eq!(extract_first_user_message(&input), "Hello world");
    }

    #[test]
    fn test_extract_first_user_message_empty() {
        let input = Input::Empty;
        assert_eq!(extract_first_user_message(&input), "");
    }

    #[test]
    fn test_extract_first_user_message_items() {
        use crate::models::MessageInput;
        let input = Input::Items(vec![InputItem::Message(MessageInput {
            role: Role::User,
            content: MessageContent::Text("Hello from items".to_string()),
        })]);
        assert_eq!(extract_first_user_message(&input), "Hello from items");
    }
}
