/**
 * Responses API Integration
 *
 * High-level functions for interacting with the streaming Responses API.
 * Uses the SSE parser from streaming.ts for event handling.
 */

import { getApiBaseUrl } from './client';
import { isCompletedEvent, isFailedEvent, isTextDeltaEvent, parseSSEStream } from './streaming';
import { logger } from '$lib/utils/logger';
import { generateUUID } from '$lib/stores/types';

/**
 * Options for streaming message requests
 */
export interface StreamingOptions {
	/** Model to use for generation */
	model?: string;
	/** Previous response ID for context chaining */
	previousResponseId?: string | null;
	/** AbortSignal for request cancellation */
	signal?: AbortSignal;
}

/**
 * Callbacks for streaming events
 */
export interface StreamingCallbacks {
	/** Called when text content is received (cumulative) */
	onDelta: (text: string) => void;
	/** Called when the response is complete */
	onComplete: (responseId: string) => void;
	/** Called when an error occurs */
	onError: (error: Error) => void;
}

/**
 * Send a message and stream the response.
 *
 * @param input - The user's message text
 * @param options - Configuration options
 * @param callbacks - Event callbacks for streaming
 *
 * @example
 * ```typescript
 * await sendMessageStreaming(
 *   'Hello, how are you?',
 *   { previousResponseId: 'resp_123' },
 *   {
 *     onDelta: (text) => console.log('Content:', text),
 *     onComplete: (id) => console.log('Done:', id),
 *     onError: (err) => console.error('Error:', err),
 *   }
 * );
 * ```
 */
export async function sendMessageStreaming(
	input: string,
	options: StreamingOptions,
	callbacks: StreamingCallbacks
): Promise<void> {
	const { model = 'gpt-oss-120b', previousResponseId = null, signal } = options;
	const { onDelta, onComplete, onError } = callbacks;

	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/responses`;

	const requestId = generateUUID().slice(0, 8);

	logger.api.request('POST', url, {
		requestId,
		model,
		inputLength: input.length,
		previousResponseId,
		stream: true
	});

	try {
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				model,
				input,
				previous_response_id: previousResponseId,
				stream: true,
				store: true
			}),
			signal
		});

		logger.api.response('POST', url, response.status, { requestId });

		if (!response.ok) {
			const errorText = await response.text();
			logger.error('api', 'API error response', { requestId, status: response.status, errorText });
			throw new Error(`API error ${response.status}: ${errorText}`);
		}

		let responseId = '';
		let fullText = '';
		let chunkCount = 0;

		for await (const event of parseSSEStream(response)) {
			// Handle text delta events
			if (isTextDeltaEvent(event)) {
				fullText += event.delta;
				chunkCount++;
				onDelta(fullText);
			}

			// Track response ID from created event
			if (event.type === 'response.created') {
				responseId = (event as { response?: { id?: string } }).response?.id ?? '';
				logger.info('streaming', 'Response created', { requestId, responseId });
			}

			// Handle completion
			if (isCompletedEvent(event)) {
				// Extract response ID from completed event if not already set
				if (!responseId && event.response?.id) {
					responseId = event.response.id;
				}
				logger.info('streaming', 'Stream completed', {
					requestId,
					responseId,
					totalLength: fullText.length,
					chunkCount
				});
				onComplete(responseId);
				return;
			}

			// Handle failure
			if (isFailedEvent(event)) {
				const error = (event as { error?: { message?: string } }).error;
				logger.error('streaming', 'Stream failed', { requestId, error });
				throw new Error(error?.message ?? 'Response failed');
			}
		}

		// Stream ended without completion event
		if (responseId) {
			logger.warn('streaming', 'Stream ended without completion event', { requestId, responseId });
			onComplete(responseId);
		}
	} catch (error) {
		// Handle abort specifically
		if (error instanceof Error && error.name === 'AbortError') {
			logger.warn('api', 'Request aborted', { requestId });
			onError(new Error('Request was cancelled'));
			return;
		}

		logger.error('api', 'Request failed', {
			requestId,
			error: error instanceof Error ? error.message : String(error)
		});
		onError(error instanceof Error ? error : new Error(String(error)));
	}
}

/**
 * Send a message without streaming (single response).
 *
 * @param input - The user's message text
 * @param options - Configuration options
 * @returns The complete response text and ID
 */
export async function sendMessage(
	input: string,
	options: StreamingOptions = {}
): Promise<{ text: string; responseId: string }> {
	const { model = 'gpt-oss-120b', previousResponseId = null, signal } = options;

	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/responses`;

	const requestId = generateUUID().slice(0, 8);

	logger.api.request('POST', url, {
		requestId,
		model,
		inputLength: input.length,
		previousResponseId,
		stream: false
	});

	const response = await fetch(url, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			model,
			input,
			previous_response_id: previousResponseId,
			stream: false,
			store: true
		}),
		signal
	});

	logger.api.response('POST', url, response.status, { requestId });

	if (!response.ok) {
		const errorText = await response.text();
		logger.error('api', 'API error response', { requestId, status: response.status, errorText });
		throw new Error(`API error ${response.status}: ${errorText}`);
	}

	const data = (await response.json()) as {
		id: string;
		output?: Array<{ type: string; content?: Array<{ type: string; text?: string }> }>;
	};

	// Extract text from response output
	let text = '';
	if (data.output) {
		for (const item of data.output) {
			if (item.type === 'message' && item.content) {
				for (const content of item.content) {
					if (content.type === 'output_text' && content.text) {
						text += content.text;
					}
				}
			}
		}
	}

	logger.info('api', 'Response received', {
		requestId,
		responseId: data.id,
		textLength: text.length
	});

	return {
		text,
		responseId: data.id
	};
}
