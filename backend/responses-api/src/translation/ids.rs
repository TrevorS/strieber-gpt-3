//! ID generation utilities for Responses API objects.
//!
//! OpenAI uses specific prefixes for different object types.
//! We maintain compatibility with these conventions.

use uuid::Uuid;

/// Generate a response ID (resp_...)
pub fn response_id() -> String {
    format!("resp_{}", short_id())
}

/// Generate a message output ID (msg_...)
pub fn message_id() -> String {
    format!("msg_{}", short_id())
}

/// Generate a function call ID (fc_...)
pub fn function_call_id() -> String {
    format!("fc_{}", short_id())
}

/// Generate a call ID for tool calls (call_...)
pub fn call_id() -> String {
    format!("call_{}", short_id())
}

/// Generate a reasoning item ID (rs_...)
pub fn reasoning_id() -> String {
    format!("rs_{}", short_id())
}

/// Generate a custom tool call ID (ctc_...)
pub fn custom_tool_call_id() -> String {
    format!("ctc_{}", short_id())
}

/// Generate a short hex ID (32 chars from UUID v4)
fn short_id() -> String {
    Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_id_has_correct_prefix() {
        let id = response_id();
        assert!(id.starts_with("resp_"), "got: {id}");
        assert_eq!(id.len(), 5 + 32); // prefix + uuid
    }

    #[test]
    fn message_id_has_correct_prefix() {
        let id = message_id();
        assert!(id.starts_with("msg_"), "got: {id}");
    }

    #[test]
    fn function_call_id_has_correct_prefix() {
        let id = function_call_id();
        assert!(id.starts_with("fc_"), "got: {id}");
    }

    #[test]
    fn call_id_has_correct_prefix() {
        let id = call_id();
        assert!(id.starts_with("call_"), "got: {id}");
    }

    #[test]
    fn ids_are_unique() {
        let id1 = response_id();
        let id2 = response_id();
        assert_ne!(id1, id2);
    }
}
