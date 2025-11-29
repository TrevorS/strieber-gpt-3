//! Server-Sent Events types for streaming responses.
//!
//! These events follow the OpenAI Responses API streaming format.

use serde::Serialize;

use super::{OutputItem, Response};

/// All possible streaming events.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Response object created
    #[serde(rename = "response.created")]
    ResponseCreated { response: Response },

    /// Response is in progress
    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: Response },

    /// New output item added
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        response_id: String,
        output_index: u32,
        item: OutputItem,
    },

    /// Output item completed
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        response_id: String,
        output_index: u32,
        item: OutputItem,
    },

    /// Text content delta
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Text content complete
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },

    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        delta: String,
    },

    /// Function call arguments complete
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        response_id: String,
        item_id: String,
        output_index: u32,
        arguments: String,
    },

    /// Reasoning content delta (for reasoning models)
    #[serde(rename = "response.reasoning.delta")]
    ReasoningDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Reasoning content complete
    #[serde(rename = "response.reasoning.done")]
    ReasoningDone {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },

    /// Reasoning summary delta
    #[serde(rename = "response.reasoning_summary.delta")]
    ReasoningSummaryDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        delta: String,
    },

    /// Reasoning summary complete
    #[serde(rename = "response.reasoning_summary.done")]
    ReasoningSummaryDone {
        response_id: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        text: String,
    },

    /// Response completed successfully
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: Response },

    /// Response failed
    #[serde(rename = "response.failed")]
    ResponseFailed { response: Response },

    /// Response done (final event)
    #[serde(rename = "response.done")]
    ResponseDone { response: Response },

    /// Error event
    Error { error: StreamError },
}

/// Error in streaming response.
#[derive(Debug, Clone, Serialize)]
pub struct StreamError {
    pub code: String,
    pub message: String,
}

/// Wrapper for SSE serialization with correct event type.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: &'static str,
    pub data: StreamEvent,
}

impl SseEvent {
    pub fn response_created(response: Response) -> Self {
        Self {
            event: "response.created",
            data: StreamEvent::ResponseCreated { response },
        }
    }

    pub fn response_in_progress(response: Response) -> Self {
        Self {
            event: "response.in_progress",
            data: StreamEvent::ResponseInProgress { response },
        }
    }

    pub fn output_item_added(response_id: String, output_index: u32, item: OutputItem) -> Self {
        Self {
            event: "response.output_item.added",
            data: StreamEvent::OutputItemAdded {
                response_id,
                output_index,
                item,
            },
        }
    }

    pub fn output_text_delta(
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    ) -> Self {
        Self {
            event: "response.output_text.delta",
            data: StreamEvent::OutputTextDelta {
                response_id,
                item_id,
                output_index,
                content_index,
                delta,
            },
        }
    }

    pub fn output_text_done(
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    ) -> Self {
        Self {
            event: "response.output_text.done",
            data: StreamEvent::OutputTextDone {
                response_id,
                item_id,
                output_index,
                content_index,
                text,
            },
        }
    }

    pub fn function_call_arguments_delta(
        response_id: String,
        item_id: String,
        output_index: u32,
        delta: String,
    ) -> Self {
        Self {
            event: "response.function_call_arguments.delta",
            data: StreamEvent::FunctionCallArgumentsDelta {
                response_id,
                item_id,
                output_index,
                delta,
            },
        }
    }

    pub fn function_call_arguments_done(
        response_id: String,
        item_id: String,
        output_index: u32,
        arguments: String,
    ) -> Self {
        Self {
            event: "response.function_call_arguments.done",
            data: StreamEvent::FunctionCallArgumentsDone {
                response_id,
                item_id,
                output_index,
                arguments,
            },
        }
    }

    pub fn output_item_done(response_id: String, output_index: u32, item: OutputItem) -> Self {
        Self {
            event: "response.output_item.done",
            data: StreamEvent::OutputItemDone {
                response_id,
                output_index,
                item,
            },
        }
    }

    pub fn response_completed(response: Response) -> Self {
        Self {
            event: "response.completed",
            data: StreamEvent::ResponseCompleted { response },
        }
    }

    pub fn response_done(response: Response) -> Self {
        Self {
            event: "response.done",
            data: StreamEvent::ResponseDone { response },
        }
    }

    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            event: "error",
            data: StreamEvent::Error {
                error: StreamError {
                    code: code.into(),
                    message: message.into(),
                },
            },
        }
    }

    pub fn reasoning_delta(
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    ) -> Self {
        Self {
            event: "response.reasoning.delta",
            data: StreamEvent::ReasoningDelta {
                response_id,
                item_id,
                output_index,
                content_index,
                delta,
            },
        }
    }

    pub fn reasoning_done(
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    ) -> Self {
        Self {
            event: "response.reasoning.done",
            data: StreamEvent::ReasoningDone {
                response_id,
                item_id,
                output_index,
                content_index,
                text,
            },
        }
    }

    pub fn reasoning_summary_delta(
        response_id: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        delta: String,
    ) -> Self {
        Self {
            event: "response.reasoning_summary.delta",
            data: StreamEvent::ReasoningSummaryDelta {
                response_id,
                item_id,
                output_index,
                summary_index,
                delta,
            },
        }
    }

    pub fn reasoning_summary_done(
        response_id: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
        text: String,
    ) -> Self {
        Self {
            event: "response.reasoning_summary.done",
            data: StreamEvent::ReasoningSummaryDone {
                response_id,
                item_id,
                output_index,
                summary_index,
                text,
            },
        }
    }
}
