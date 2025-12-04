/**
 * Responses API Integration
 *
 * High-level functions for interacting with the streaming Responses API.
 * Uses the SSE parser from streaming.ts for event handling.
 */

import { getApiBaseUrl } from './client';
import { isCompletedEvent, isFailedEvent, isTextDeltaEvent, parseSSEStream } from './streaming';
import type { ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent } from './streaming';
import { logger } from '$lib/utils/logger';
import { generateUUID, type ResponseOutputItem } from '$lib/stores/types';
import { formatTextAttachmentsForPrompt, type Attachment } from '$lib/utils/files';

/**
 * Tool definition for API requests
 */
export interface ToolDefinition {
	type: string;
	[key: string]: unknown;
}

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
	/** Tools to enable for this request */
	tools?: ToolDefinition[];
	/** File attachments to include with the message */
	attachments?: Attachment[];
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
	/** Called when an output item is added (optional) */
	onOutputItem?: (item: ResponseOutputItem, status: 'added' | 'done') => void;
	/** Called when reasoning text is received (cumulative) - for DeepSeek R1 / o-series models */
	onReasoning?: (text: string) => void;
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
/**
 * Format message input with attachments.
 * - Text files are prepended as demarcated code blocks
 * - Images use multimodal content format
 */
type ContentPart = { type: string; text?: string; image_url?: { url: string; detail?: string } };
type MessageInput = { type: 'message'; role: 'user'; content: ContentPart[] };

function formatInputWithAttachments(
	text: string,
	attachments: Attachment[]
): string | MessageInput[] {
	const imageAttachments = attachments.filter((a) => a.type === 'image');
	const textContent = formatTextAttachmentsForPrompt(attachments);

	// Combine text attachments with user message
	const fullText = textContent ? `${textContent}\n\n${text}` : text;

	// If no images, return plain string
	if (imageAttachments.length === 0) {
		return fullText;
	}

	// With images, use multimodal content array wrapped in a message object
	const parts: Array<{
		type: string;
		text?: string;
		image_url?: { url: string; detail?: string };
	}> = [];

	// Add text first
	if (fullText) {
		parts.push({ type: 'input_text', text: fullText });
	}

	// Add images
	for (const img of imageAttachments) {
		parts.push({
			type: 'input_image',
			image_url: {
				url: img.content,
				detail: 'auto'
			}
		});
	}

	// Wrap content parts in a message object (backend expects InputItem with type: "message")
	return [
		{
			type: 'message',
			role: 'user',
			content: parts
		}
	];
}

export async function sendMessageStreaming(
	input: string,
	options: StreamingOptions,
	callbacks: StreamingCallbacks
): Promise<void> {
	const {
		model = 'gpt-oss-120b',
		previousResponseId = null,
		signal,
		tools = [],
		attachments = []
	} = options;
	const { onDelta, onComplete, onError, onOutputItem, onReasoning } = callbacks;

	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/responses`;

	const requestId = generateUUID().slice(0, 8);

	// Format input with any attachments
	const formattedInput = formatInputWithAttachments(input, attachments);
	const hasImages = attachments.some((a) => a.type === 'image');

	logger.api.request('POST', url, {
		requestId,
		model,
		inputLength: input.length,
		previousResponseId,
		tools: tools.length,
		attachments: attachments.length,
		hasImages,
		stream: true
	});

	// Extra debug logging for context chain investigation
	logger.info('api', '=== CONTEXT CHAIN DEBUG ===', {
		requestId,
		previousResponseId: previousResponseId ?? 'null (new conversation)',
		inputPreview: input.length > 50 ? `${input.slice(0, 50)}...` : input,
		attachments: attachments.map((a) => ({ name: a.name, type: a.type }))
	});

	try {
		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				model,
				input: formattedInput,
				previous_response_id: previousResponseId,
				stream: true,
				store: true,
				tools
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
		let reasoningText = '';
		let chunkCount = 0;

		for await (const event of parseSSEStream(response)) {
			// Handle text delta events
			if (isTextDeltaEvent(event)) {
				fullText += event.delta;
				chunkCount++;
				onDelta(fullText);
			}

			// Handle reasoning delta events (o-series, DeepSeek R1)
			// Use string comparison since SDK types may not include all event types
			if ((event as { type: string }).type === 'response.reasoning_text.delta' && onReasoning) {
				const delta = (event as { delta?: string }).delta ?? '';
				reasoningText += delta;
				onReasoning(reasoningText);
			}

			// Handle output item added events
			if (event.type === 'response.output_item.added' && onOutputItem) {
				const addedEvent = event as ResponseOutputItemAddedEvent;
				logger.debug('streaming', 'Output item added', {
					requestId,
					itemType: addedEvent.item.type,
					outputIndex: addedEvent.output_index
				});
				onOutputItem(addedEvent.item as ResponseOutputItem, 'added');
			}

			// Handle output item done events
			if (event.type === 'response.output_item.done' && onOutputItem) {
				const doneEvent = event as ResponseOutputItemDoneEvent;
				logger.debug('streaming', 'Output item done', {
					requestId,
					itemType: doneEvent.item.type,
					outputIndex: doneEvent.output_index
				});
				onOutputItem(doneEvent.item as ResponseOutputItem, 'done');
			}

			// Track response ID from created event
			if (event.type === 'response.created') {
				responseId = (event as { response?: { id?: string } }).response?.id ?? '';
				logger.info('streaming', 'Response created', { requestId, responseId });
				logger.info('api', '=== RESPONSE ID RECEIVED ===', {
					requestId,
					responseId,
					previousResponseId: previousResponseId ?? 'null',
					note: 'This responseId will become previousResponseId for next message'
				});
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
					reasoningLength: reasoningText.length,
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
