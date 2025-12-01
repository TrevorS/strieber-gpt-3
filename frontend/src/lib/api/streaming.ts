/**
 * SSE Stream Parser for Responses API
 *
 * Parses Server-Sent Events from the Responses API streaming endpoint.
 * Handles all event types, [DONE] terminator, and connection errors.
 */

// Re-export the parsed event type from OpenAI
export type { ParsedResponseStreamEvent as StreamEvent } from 'openai/lib/responses/EventTypes';

// Re-export individual event types for convenience
export type {
	ResponseCreatedEvent,
	ResponseInProgressEvent,
	ResponseCompletedEvent,
	ResponseFailedEvent,
	ResponseIncompleteEvent,
	ResponseErrorEvent,
	ResponseOutputItemAddedEvent,
	ResponseOutputItemDoneEvent,
	ResponseContentPartAddedEvent,
	ResponseContentPartDoneEvent,
	ResponseTextDoneEvent,
	ResponseRefusalDeltaEvent,
	ResponseRefusalDoneEvent,
	ResponseFunctionCallArgumentsDoneEvent,
	ResponseFileSearchCallInProgressEvent,
	ResponseFileSearchCallSearchingEvent,
	ResponseFileSearchCallCompletedEvent,
	ResponseWebSearchCallInProgressEvent,
	ResponseWebSearchCallSearchingEvent,
	ResponseWebSearchCallCompletedEvent,
	ResponseCodeInterpreterCallInProgressEvent,
	ResponseCodeInterpreterCallInterpretingEvent,
	ResponseCodeInterpreterCallCodeDoneEvent,
	ResponseCodeInterpreterCallCompletedEvent
} from 'openai/resources/responses/responses';

// These have snapshot fields added by the SDK
export type {
	ResponseTextDeltaEvent,
	ResponseFunctionCallArgumentsDeltaEvent
} from 'openai/lib/responses/EventTypes';

/**
 * Custom error for stream parsing failures
 */
export class StreamParseError extends Error {
	constructor(
		message: string,
		public readonly cause?: unknown
	) {
		super(message);
		this.name = 'StreamParseError';
	}
}

/**
 * Parse a single SSE data line into a typed event.
 *
 * @param data - The data portion of an SSE event (after "data: ")
 * @returns Parsed event object, or null for [DONE] terminator
 * @throws StreamParseError if JSON parsing fails
 */
export function parseSSEData(
	data: string
): import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent | null {
	if (data === '[DONE]') {
		return null;
	}

	try {
		return JSON.parse(data);
	} catch (error) {
		throw new StreamParseError(`Failed to parse SSE data: ${data}`, error);
	}
}

/**
 * Parse an SSE stream from a fetch Response.
 *
 * Yields typed events as they arrive. Handles:
 * - Line buffering for partial chunks
 * - [DONE] terminator detection
 * - Empty lines and comments (lines starting with :)
 * - Both "data:" and "event:" fields (though we only use data)
 *
 * @param response - Fetch Response with SSE body
 * @yields Parsed stream events
 * @throws StreamParseError on parsing errors
 *
 * @example
 * ```typescript
 * const response = await fetch('/v1/responses', { method: 'POST', body: ... });
 * for await (const event of parseSSEStream(response)) {
 *   if (event.type === 'response.output_text.delta') {
 *     console.log(event.delta);
 *   }
 * }
 * ```
 */
export async function* parseSSEStream(
	response: globalThis.Response
): AsyncGenerator<import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent> {
	if (!response.body) {
		throw new StreamParseError('Response body is null');
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	try {
		while (true) {
			const { done, value } = await reader.read();

			if (done) {
				// Process any remaining buffer content
				if (buffer.trim()) {
					const event = processLine(buffer);
					if (event) yield event;
				}
				break;
			}

			// Decode chunk and add to buffer
			buffer += decoder.decode(value, { stream: true });

			// Process complete lines
			const lines = buffer.split('\n');
			// Keep the last (potentially incomplete) line in the buffer
			buffer = lines.pop() ?? '';

			for (const line of lines) {
				const event = processLine(line);
				if (event === null) {
					// [DONE] received, stop iteration
					return;
				}
				if (event) {
					yield event;
				}
			}
		}
	} finally {
		reader.releaseLock();
	}
}

/**
 * Process a single line from the SSE stream.
 *
 * @param line - A complete line from the stream
 * @returns Parsed event, null for [DONE], or undefined to skip
 */
function processLine(
	line: string
): import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent | null | undefined {
	const trimmed = line.trim();

	// Skip empty lines and comments
	if (!trimmed || trimmed.startsWith(':')) {
		return undefined;
	}

	// Handle data lines
	if (trimmed.startsWith('data:')) {
		const data = trimmed.slice(5).trim();
		return parseSSEData(data);
	}

	// Skip event type lines (we determine type from data.type)
	if (trimmed.startsWith('event:')) {
		return undefined;
	}

	// Unknown line format, skip
	return undefined;
}

/**
 * Helper to check if an event is a text delta event.
 */
export function isTextDeltaEvent(
	event: import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent
): event is import('openai/lib/responses/EventTypes').ResponseTextDeltaEvent {
	return event.type === 'response.output_text.delta';
}

/**
 * Helper to check if an event is a completion event.
 */
export function isCompletedEvent(
	event: import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent
): event is import('openai/resources/responses/responses').ResponseCompletedEvent {
	return event.type === 'response.completed';
}

/**
 * Helper to check if an event is a failure event.
 */
export function isFailedEvent(
	event: import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent
): event is import('openai/resources/responses/responses').ResponseFailedEvent {
	return event.type === 'response.failed';
}

/**
 * Helper to check if an event is an error event.
 */
export function isErrorEvent(
	event: import('openai/lib/responses/EventTypes').ParsedResponseStreamEvent
): event is import('openai/resources/responses/responses').ResponseErrorEvent {
	return event.type === 'error';
}
