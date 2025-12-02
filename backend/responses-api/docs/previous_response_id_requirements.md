# Requirements: `previous_response_id` Backend Implementation

## Overview

The `previous_response_id` field enables multi-turn conversations by chaining responses together. When a client provides this field, the backend must fetch the previous response from storage and reconstruct the full conversation context before calling the LLM.

**Current State**: The field is accepted in requests and echoed back in responses, but no actual context chaining occurs.

---

## Functional Requirements

### FR-1: Resolve Previous Response

**Description**: When a request includes `previous_response_id`, the system must fetch the stored response and its original request from the response store.

**Acceptance Criteria**:
- Given a request with `previous_response_id = "resp_abc123"`
- When the system processes the request
- Then it fetches the stored response with ID `"resp_abc123"` from the response store
- And it has access to both the `Response` object and the original `CreateResponseRequest`

**Error Cases**:
- If `previous_response_id` is provided but the response doesn't exist in storage, return a 400 error with message: `"previous_response_id 'resp_xxx' not found"`
- If `previous_response_id` is provided but the stored response has `status: "failed"`, return a 400 error with message: `"cannot chain from failed response 'resp_xxx'"`

---

### FR-2: Recursive Context Resolution

**Description**: Previous responses may themselves have a `previous_response_id`, creating a chain. The system must recursively resolve the entire chain to build complete conversation context.

**Acceptance Criteria**:
- Given Response A → Response B → Response C (current request)
- When processing Request C with `previous_response_id = B.id`
- Then the system resolves B, discovers B has `previous_response_id = A.id`
- And resolves A (which has no previous)
- And builds context in order: [A's input, A's output, B's input, B's output, C's input]

**Constraints**:
- Maximum chain depth: 100 responses (configurable)
- If chain depth exceeded, return 400 error: `"conversation chain exceeds maximum depth of 100"`

---

### FR-3: Output-to-Input Conversion

**Description**: Previous response outputs must be converted to input items to be included in the conversation context for the LLM.

**Conversion Rules**:

| Output Type | Converts To | Notes |
|-------------|-------------|-------|
| `OutputItem::Message` (role: assistant) | `InputItem::Message` (role: assistant) | Copy content directly |
| `OutputItem::Reasoning` | `InputItem::Reasoning` | Copy reasoning content |
| `OutputItem::FunctionCall` | `InputItem::FunctionCall` | Copy call_id, name, arguments |
| `OutputItem::WebSearchCall` | (skip) | Built-in tool, results already in context |
| `OutputItem::CodeInterpreterCall` | (skip) | Built-in tool, results already in context |
| `OutputItem::FileSearchCall` | (skip) | Built-in tool, results already in context |
| `OutputItem::ComputerCall` | (skip) | Built-in tool, results already in context |
| `OutputItem::CustomToolCall` | `InputItem::CustomToolCallOutput` | Represents the tool result |

**Acceptance Criteria**:
- Given a previous response with output: `[Reasoning, Message, FunctionCall]`
- When converting to input items
- Then the result is: `[Reasoning (as input), Message (as input), FunctionCall (as input)]`
- And built-in tool calls (web_search, code_interpreter, etc.) are excluded

---

### FR-4: Context Assembly Order

**Description**: The conversation context must be assembled in chronological order for proper LLM understanding.

**Assembly Order**:
1. System instructions (from current request, or inherited from chain if not specified)
2. Previous request's input items (converted to chat messages)
3. Previous response's output items (converted to input items, then to chat messages)
4. (Repeat 2-3 for each link in the chain)
5. Current request's input items

**Acceptance Criteria**:
- Given: Request A (input: "Hello") → Response A (output: "Hi there!") → Request B (input: "How are you?", previous_response_id: A.id)
- When building chat messages for LLM
- Then the order is:
  1. System message (if any)
  2. User: "Hello"
  3. Assistant: "Hi there!"
  4. User: "How are you?"

---

### FR-5: Instruction Inheritance

**Description**: If the current request doesn't specify `instructions` (system prompt), inherit from the most recent request in the chain that has one.

**Acceptance Criteria**:
- Given Request A with `instructions: "You are a helpful assistant"`
- And Request B with no `instructions`, `previous_response_id: A.id`
- When processing Request B
- Then use `"You are a helpful assistant"` as the system prompt

- Given Request A with `instructions: "You are helpful"`
- And Request B with `instructions: "You are funny"`, `previous_response_id: A.id`
- When processing Request B
- Then use `"You are funny"` (current request takes precedence)

---

### FR-6: Streaming Support

**Description**: The `previous_response_id` resolution must work identically for both streaming and non-streaming requests.

**Acceptance Criteria**:
- Given a streaming request with `previous_response_id`
- When processing via SSE
- Then the same context resolution and assembly occurs before streaming begins
- And the `response.created` event includes the resolved `previous_response_id`

---

### FR-7: No Tool Inheritance (OpenAI Behavior)

**Description**: Tools are NOT inherited when using `previous_response_id`. Each request must explicitly specify its own tools.

**Reference**: Per [Azure OpenAI Responses API documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses), function calling examples show that tools must be explicitly redefined in subsequent requests—they are not automatically inherited from the previous response.

**Acceptance Criteria**:
- Given Request A with `tools: [weather_tool]`
- And Request B with no `tools`, `previous_response_id: A.id`
- When processing Request B
- Then Request B has NO tools available (not inherited from A)
- And if the model attempts a tool call, it will fail or be ignored

**Rationale**: This matches OpenAI's behavior and gives clients explicit control over tool availability per request.

---

## Non-Functional Requirements

### NFR-1: Performance

**Description**: Context resolution should not significantly impact response latency.

**Acceptance Criteria**:
- Single-hop resolution (one previous response): < 5ms additional latency
- Full chain resolution (10 responses): < 50ms additional latency
- Resolution is O(n) where n = chain length

---

### NFR-2: Memory Efficiency

**Description**: Context assembly should not duplicate large amounts of data unnecessarily.

**Acceptance Criteria**:
- Use references where possible during assembly
- Only clone data when necessary for the final chat messages
- Large attachments (images, files) in previous responses should be handled carefully

---

### NFR-3: Error Handling

**Description**: Clear, actionable error messages for all failure modes.

**Error Responses** (HTTP 400):
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "previous_response_id 'resp_xxx' not found"
  }
}
```

**Error Cases**:
- Response not found in store
- Response has failed status
- Chain depth exceeded
- Circular reference detected (response chains to itself or creates a loop)

---

### NFR-4: Observability

**Description**: Chain resolution should be observable for debugging.

**Acceptance Criteria**:
- Log chain resolution at DEBUG level: `"Resolving chain: [resp_c] → [resp_b] → [resp_a]"`
- Include chain length in response metadata or headers (optional)

---

## Technical Constraints

### TC-1: Store Dependency

The executor currently doesn't have access to the response store. The store must be passed to the executor, or resolution must happen in the handler before calling the executor.

**Recommended Approach**: Resolve in the handler, pass assembled context to executor.

---

### TC-2: TTL Considerations

Stored responses have a TTL. Current implementation uses 1 hour default, but OpenAI's Responses API retains responses for 30 days. Consider making this configurable.

**Behavior**: Treat expired responses the same as non-existent (return 400 error with "previous_response_id not found").

**Note**: Per [OpenAI community discussions](https://community.openai.com/t/how-long-do-previous-messages-in-the-previous-response-id-last/1280341), responses last 30 days in OpenAI's implementation.

---

### TC-3: Existing Translation Functions

The codebase has `translation/request.rs` with `to_chat_completion()` which already accepts a `previous_messages` parameter. This is the integration point.

---

## Requirement Dependencies

```
FR-1 (Resolve Previous) ──────┐
                              │
FR-2 (Recursive Resolution) ──┼──→ FR-4 (Context Assembly) ──→ FR-6 (Streaming)
                              │
FR-3 (Output-to-Input) ───────┘

FR-5 (Instruction Inheritance) ──→ FR-4 (Context Assembly)

FR-7 (Tool Inheritance) ──→ Optional, can be implemented later
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/server/handlers.rs` | Pass store to executor or resolve chain before execute() |
| `src/execution/executor.rs` | Accept previous context, use in execute() |
| `src/execution/streaming.rs` | Same changes for streaming path |
| `src/translation/request.rs` | Add `response_output_to_input_items()` function |
| `src/models/input.rs` | Possibly add helper constructors |
| `tests/` | Add integration tests for multi-turn conversations |

---

## Test Cases

### Happy Path
1. Two-turn conversation: Request A → Response A → Request B (with previous_response_id) → Response B
2. Three-turn conversation with tool calls in the middle
3. Chain with reasoning output included

### Error Cases
4. `previous_response_id` references non-existent response
5. `previous_response_id` references failed response
6. Chain exceeds maximum depth
7. Circular reference (if technically possible)

### Edge Cases
8. Empty input in current request (just continuing from previous)
9. Previous response had no output (empty assistant response)
10. Mixed content types (text + images) in chain
