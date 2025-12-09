//! ID generation for Responses API objects.
//!
//! OpenAI uses specific prefixes for different object types.

use uuid::Uuid;

pub fn response_id() -> String {
    format!("resp_{}", short_id())
}

pub fn message_id() -> String {
    format!("msg_{}", short_id())
}

pub fn function_call_id() -> String {
    format!("fc_{}", short_id())
}

/// For reasoning model support.
pub fn reasoning_id() -> String {
    format!("rs_{}", short_id())
}

/// Generate a conversation ID (conv_<uuid>)
pub fn conversation_id() -> String {
    format!("conv_{}", short_id())
}

/// Generate a generic item ID for conversation items (item_<uuid>)
pub fn item_id() -> String {
    format!("item_{}", short_id())
}

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
    fn ids_are_unique() {
        let id1 = response_id();
        let id2 = response_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn conversation_id_has_correct_prefix() {
        let id = conversation_id();
        assert!(id.starts_with("conv_"), "got: {id}");
        assert_eq!(id.len(), 5 + 32); // prefix + uuid
    }

    #[test]
    fn item_id_has_correct_prefix() {
        let id = item_id();
        assert!(id.starts_with("item_"), "got: {id}");
        assert_eq!(id.len(), 5 + 32); // prefix + uuid
    }
}
