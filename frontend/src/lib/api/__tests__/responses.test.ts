/**
 * Unit tests for responses API functions
 *
 * Tests cover:
 * - sendMessageStreaming: delta handling, completion, errors
 * - sendMessage: non-streaming requests
 * - Request cancellation via AbortSignal
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { sendMessage, sendMessageStreaming } from '../responses';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Helper to create a mock SSE response
function createSSEResponse(events: string[]): Response {
	const body = `${events.join('\n\n')}\n\n`;
	const encoder = new TextEncoder();
	const stream = new ReadableStream({
		start(controller) {
			controller.enqueue(encoder.encode(body));
			controller.close();
		}
	});

	return new Response(stream, {
		status: 200,
		headers: { 'Content-Type': 'text/event-stream' }
	});
}

// Helper to create a mock JSON response
function createJSONResponse(data: unknown, status = 200): Response {
	return new Response(JSON.stringify(data), {
		status,
		headers: { 'Content-Type': 'application/json' }
	});
}

describe('sendMessageStreaming', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	// ============================================================================
	// Basic Streaming
	// ============================================================================

	describe('basic streaming', () => {
		it('should call onDelta with accumulated text', async () => {
			const events = [
				'data: {"type":"response.created","response":{"id":"resp_123"}}',
				'data: {"type":"response.output_text.delta","delta":"Hello"}',
				'data: {"type":"response.output_text.delta","delta":" World"}',
				'data: {"type":"response.completed","response":{"id":"resp_123"}}'
			];

			mockFetch.mockResolvedValueOnce(createSSEResponse(events));

			const deltas: string[] = [];
			const onDelta = vi.fn((text: string) => deltas.push(text));
			const onComplete = vi.fn();
			const onError = vi.fn();

			await sendMessageStreaming('test', {}, { onDelta, onComplete, onError });

			expect(onDelta).toHaveBeenCalledTimes(2);
			expect(deltas).toEqual(['Hello', 'Hello World']);
			expect(onComplete).toHaveBeenCalledWith('resp_123');
			expect(onError).not.toHaveBeenCalled();
		});

		it('should pass correct request body', async () => {
			const events = [
				'data: {"type":"response.created","response":{"id":"resp_123"}}',
				'data: {"type":"response.completed","response":{"id":"resp_123"}}'
			];

			mockFetch.mockResolvedValueOnce(createSSEResponse(events));

			await sendMessageStreaming(
				'Hello',
				{ model: 'test-model', conversationId: 'conv_123' },
				{ onDelta: vi.fn(), onComplete: vi.fn(), onError: vi.fn() }
			);

			expect(mockFetch).toHaveBeenCalledWith(
				expect.stringContaining('/responses'),
				expect.objectContaining({
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						model: 'test-model',
						input: 'Hello',
						conversation: { id: 'conv_123' },
						stream: true,
						store: true,
						tools: []
					})
				})
			);
		});

		it('should use default model when not specified', async () => {
			const events = [
				'data: {"type":"response.created","response":{"id":"resp_123"}}',
				'data: {"type":"response.completed","response":{"id":"resp_123"}}'
			];

			mockFetch.mockResolvedValueOnce(createSSEResponse(events));

			await sendMessageStreaming(
				'Hello',
				{},
				{ onDelta: vi.fn(), onComplete: vi.fn(), onError: vi.fn() }
			);

			const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
			expect(callBody.model).toBe('gpt-oss-120b');
		});
	});

	// ============================================================================
	// Error Handling
	// ============================================================================

	describe('error handling', () => {
		it('should call onError for HTTP errors', async () => {
			mockFetch.mockResolvedValueOnce(new Response('Internal Server Error', { status: 500 }));

			const onError = vi.fn();

			await sendMessageStreaming('test', {}, { onDelta: vi.fn(), onComplete: vi.fn(), onError });

			expect(onError).toHaveBeenCalled();
			expect(onError.mock.calls[0][0].message).toContain('API error 500');
		});

		it('should call onError for failed events', async () => {
			const events = [
				'data: {"type":"response.created","response":{"id":"resp_123"}}',
				'data: {"type":"response.failed","error":{"message":"Something went wrong"}}'
			];

			mockFetch.mockResolvedValueOnce(createSSEResponse(events));

			const onError = vi.fn();

			await sendMessageStreaming('test', {}, { onDelta: vi.fn(), onComplete: vi.fn(), onError });

			expect(onError).toHaveBeenCalled();
			expect(onError.mock.calls[0][0].message).toBe('Something went wrong');
		});

		it('should handle network errors', async () => {
			mockFetch.mockRejectedValueOnce(new Error('Network error'));

			const onError = vi.fn();

			await sendMessageStreaming('test', {}, { onDelta: vi.fn(), onComplete: vi.fn(), onError });

			expect(onError).toHaveBeenCalled();
			expect(onError.mock.calls[0][0].message).toBe('Network error');
		});
	});

	// ============================================================================
	// Cancellation
	// ============================================================================

	describe('cancellation', () => {
		it('should pass abort signal to fetch', async () => {
			const events = [
				'data: {"type":"response.created","response":{"id":"resp_123"}}',
				'data: {"type":"response.completed","response":{"id":"resp_123"}}'
			];

			mockFetch.mockResolvedValueOnce(createSSEResponse(events));

			const controller = new AbortController();

			await sendMessageStreaming(
				'test',
				{ signal: controller.signal },
				{ onDelta: vi.fn(), onComplete: vi.fn(), onError: vi.fn() }
			);

			// Verify signal was passed to fetch
			expect(mockFetch).toHaveBeenCalledWith(
				expect.any(String),
				expect.objectContaining({
					signal: controller.signal
				})
			);
		});

		it('should handle aborted fetch', async () => {
			const abortError = new Error('Aborted');
			abortError.name = 'AbortError';
			mockFetch.mockRejectedValueOnce(abortError);

			const onError = vi.fn();

			await sendMessageStreaming('test', {}, { onDelta: vi.fn(), onComplete: vi.fn(), onError });

			expect(onError).toHaveBeenCalled();
			expect(onError.mock.calls[0][0].message).toBe('Request was cancelled');
		});
	});
});

describe('sendMessage', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should return text and responseId', async () => {
		const responseData = {
			id: 'resp_456',
			output: [
				{
					type: 'message',
					content: [{ type: 'output_text', text: 'Hello there!' }]
				}
			]
		};

		mockFetch.mockResolvedValueOnce(createJSONResponse(responseData));

		const result = await sendMessage('Hi');

		expect(result).toEqual({
			text: 'Hello there!',
			responseId: 'resp_456'
		});
	});

	it('should pass stream: false in request', async () => {
		mockFetch.mockResolvedValueOnce(createJSONResponse({ id: 'resp_123', output: [] }));

		await sendMessage('Hello');

		const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
		expect(callBody.stream).toBe(false);
	});

	it('should throw on HTTP errors', async () => {
		mockFetch.mockResolvedValueOnce(new Response('Not Found', { status: 404 }));

		await expect(sendMessage('test')).rejects.toThrow('API error 404');
	});
});
