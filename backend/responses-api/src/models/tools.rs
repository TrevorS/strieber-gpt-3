//! Tool definitions for the Responses API.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// A tool definition that can be passed to the model.
///
/// Supports two forms:
/// - `{"type": "function", "name": "...", ...}` - Full function definition (client-side)
/// - `{"type": "weather"}` - Built-in server-side tool (maps to MCP server)
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Tool {
    /// Full function definition (client-side tools)
    Function(FunctionToolWrapper),
    /// Built-in server-side tool (maps to MCP server by type)
    Builtin(BuiltinTool),
}

impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // First deserialize as raw Value to inspect the type
        let value = Value::deserialize(deserializer)?;

        // Check the "type" field
        let tool_type = value
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;

        if tool_type == "function" {
            // Full function definition
            let wrapper: FunctionToolWrapper =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Tool::Function(wrapper))
        } else {
            // Built-in tool type (weather, web_search, code_interpreter, etc.)
            Ok(Tool::Builtin(BuiltinTool {
                tool_type: tool_type.to_string(),
            }))
        }
    }
}

/// Wrapper for function tool with type tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolWrapper {
    #[serde(rename = "type")]
    pub tool_type: String, // Always "function"
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

/// A function tool with JSON schema parameters (without type tag).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

impl From<FunctionTool> for FunctionToolWrapper {
    fn from(f: FunctionTool) -> Self {
        Self {
            tool_type: "function".to_string(),
            name: f.name,
            description: f.description,
            parameters: f.parameters,
            strict: f.strict,
        }
    }
}

impl From<FunctionToolWrapper> for FunctionTool {
    fn from(f: FunctionToolWrapper) -> Self {
        Self {
            name: f.name,
            description: f.description,
            parameters: f.parameters,
            strict: f.strict,
        }
    }
}

/// Built-in server-side tool that maps to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "weather", "web_search", "code_interpreter", etc.
}

/// How the model should select tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A string mode: "auto", "required", or "none"
    Mode(ToolChoiceMode),
    /// Force a specific function
    Specific(SpecificToolChoice),
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Mode(ToolChoiceMode::Auto)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    #[default]
    Auto,
    Required,
    None,
}

/// Force the model to call a specific function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecificToolChoice {
    #[serde(rename = "type")]
    pub tool_type: SpecificToolType,
    pub name: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpecificToolType {
    Function,
}
