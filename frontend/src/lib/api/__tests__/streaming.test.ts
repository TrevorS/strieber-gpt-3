/**
 * Unit tests for SSE Stream Parser
 *
 * Tests cover:
 * - parseSSEData: Basic parsing, [DONE] detection, error handling
 * - parseSSEStream: Full stream parsing with mock responses
 * - Type guards: isTextDeltaEvent, isCompletedEvent, etc.
 * - Edge cases: Partial line buffering, empty lines, comments
 */

import { describe, it, expect } from 'vitest';
import {
	parseSSEData,
	parseSSEStream,
	StreamParseError,
	isTextDeltaEvent,
	isCompletedEvent,
	isFailedEvent,
	isErrorEvent
} from '../streaming';

// ============================================================================
// Helper: Create a mock ReadableStream from string chunks
// ============================================================================

function createMockResponse(chunks: string[]): globalThis.Response {
	const encoder = new TextEncoder();
	let chunkIndex = 0;

	const stream = new ReadableStream<Uint8Array>({
		pull(controller) {
			if (chunkIndex < chunks.length) {
				controller.enqueue(encoder.encode(chunks[chunkIndex]));
				chunkIndex++;
			} else {
				controller.close();
			}
		}
	});

	return new Response(stream);
}

// ============================================================================
// Test: parseSSEData
// ============================================================================

describe('parseSSEData', () => {
	it('should parse a valid JSON event', () => {
		const data = JSON.stringify({
			type: 'response.created',
			response: { id: 'resp_123' },
			sequence_number: 0
		});

		const result = parseSSEData(data);

		expect(result).toEqual({
			type: 'response.created',
			response: { id: 'resp_123' },
			sequence_number: 0
		});
	});

	it('should return null for [DONE] terminator', () => {
		const result = parseSSEData('[DONE]');
		expect(result).toBeNull();
	});

	it('should throw StreamParseError for invalid JSON', () => {
		expect(() => parseSSEData('not valid json')).toThrow(StreamParseError);
		expect(() => parseSSEData('{')).toThrow(StreamParseError);
	});

	it('should include the invalid data in error message', () => {
		try {
			parseSSEData('bad data');
			expect.fail('Should have thrown');
		} catch (error) {
			expect(error).toBeInstanceOf(StreamParseError);
			expect((error as StreamParseError).message).toContain('bad data');
		}
	});
});

// ============================================================================
// Test: parseSSEStream
// ============================================================================

describe('parseSSEStream', () => {
	it('should parse a simple stream with one event', async () => {
		const event = {
			type: 'response.created',
			response: { id: 'resp_123' },
			sequence_number: 0
		};
		const response = createMockResponse([`data: ${JSON.stringify(event)}\n\n`]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect(events[0]).toEqual(event);
	});

	it('should parse multiple events in sequence', async () => {
		const event1 = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };
		const event2 = {
			type: 'response.output_text.delta',
			delta: 'Hello',
			output_index: 0,
			content_index: 0,
			item_id: 'item_1',
			sequence_number: 1
		};
		const event3 = { type: 'response.completed', response: { id: 'resp_123' }, sequence_number: 2 };

		const response = createMockResponse([
			`data: ${JSON.stringify(event1)}\n\n`,
			`data: ${JSON.stringify(event2)}\n\n`,
			`data: ${JSON.stringify(event3)}\n\n`
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(3);
		expect(events[0].type).toBe('response.created');
		expect(events[1].type).toBe('response.output_text.delta');
		expect(events[2].type).toBe('response.completed');
	});

	it('should stop on [DONE] terminator', async () => {
		const event1 = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };
		const event2 = { type: 'response.completed', response: { id: 'resp_123' }, sequence_number: 1 };

		const response = createMockResponse([
			`data: ${JSON.stringify(event1)}\n\n`,
			`data: ${JSON.stringify(event2)}\n\n`,
			`data: [DONE]\n\n`,
			// This should NOT be yielded
			`data: ${JSON.stringify({ type: 'after.done' })}\n\n`
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(2);
		expect(events.find((e) => e.type === 'after.done')).toBeUndefined();
	});

	it('should handle partial line buffering across chunks', async () => {
		const event = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };
		const fullData = `data: ${JSON.stringify(event)}\n\n`;

		// Split the data across multiple chunks at arbitrary points
		const response = createMockResponse([
			fullData.slice(0, 10),
			fullData.slice(10, 25),
			fullData.slice(25)
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect(events[0]).toEqual(event);
	});

	it('should skip empty lines', async () => {
		const event = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };

		const response = createMockResponse([
			'\n\n\n',
			`data: ${JSON.stringify(event)}\n`,
			'\n\n',
			'data: [DONE]\n\n'
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
	});

	it('should skip comment lines (starting with :)', async () => {
		const event = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };

		const response = createMockResponse([
			': this is a comment\n',
			`:keep-alive\n`,
			`data: ${JSON.stringify(event)}\n\n`
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect(events[0].type).toBe('response.created');
	});

	it('should skip event: lines (we use data.type instead)', async () => {
		const event = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };

		const response = createMockResponse([
			'event: response.created\n',
			`data: ${JSON.stringify(event)}\n\n`
		]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
	});

	it('should throw StreamParseError for null body', async () => {
		const response = new Response(null);

		await expect(async () => {
			for await (const _ of parseSSEStream(response)) {
				// Should throw before yielding
			}
		}).rejects.toThrow(StreamParseError);
	});

	it('should handle text delta events with all fields', async () => {
		const deltaEvent = {
			type: 'response.output_text.delta',
			delta: 'Hello, world!',
			output_index: 0,
			content_index: 0,
			item_id: 'item_abc',
			logprobs: [],
			sequence_number: 5
		};

		const response = createMockResponse([`data: ${JSON.stringify(deltaEvent)}\n\n`]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect(events[0]).toMatchObject({
			type: 'response.output_text.delta',
			delta: 'Hello, world!',
			output_index: 0
		});
	});
});

// ============================================================================
// Test: Type Guards
// ============================================================================

describe('type guards', () => {
	describe('isTextDeltaEvent', () => {
		it('should return true for text delta events', () => {
			const event = {
				type: 'response.output_text.delta' as const,
				delta: 'test',
				output_index: 0,
				content_index: 0,
				item_id: 'item_1',
				logprobs: [],
				sequence_number: 1
			};
			expect(isTextDeltaEvent(event)).toBe(true);
		});

		it('should return false for other event types', () => {
			const event = {
				type: 'response.created' as const,
				response: { id: 'resp_123' },
				sequence_number: 0
			};
			expect(isTextDeltaEvent(event as any)).toBe(false);
		});
	});

	describe('isCompletedEvent', () => {
		it('should return true for completed events', () => {
			const event = {
				type: 'response.completed' as const,
				response: { id: 'resp_123' },
				sequence_number: 10
			};
			expect(isCompletedEvent(event)).toBe(true);
		});

		it('should return false for other event types', () => {
			const event = {
				type: 'response.created' as const,
				response: { id: 'resp_123' },
				sequence_number: 0
			};
			expect(isCompletedEvent(event as any)).toBe(false);
		});
	});

	describe('isFailedEvent', () => {
		it('should return true for failed events', () => {
			const event = {
				type: 'response.failed' as const,
				response: { id: 'resp_123' },
				sequence_number: 5
			};
			expect(isFailedEvent(event)).toBe(true);
		});

		it('should return false for completed events', () => {
			const event = {
				type: 'response.completed' as const,
				response: { id: 'resp_123' },
				sequence_number: 10
			};
			expect(isFailedEvent(event as any)).toBe(false);
		});
	});

	describe('isErrorEvent', () => {
		it('should return true for error events', () => {
			const event = {
				type: 'error' as const,
				code: 'server_error',
				message: 'Something went wrong',
				sequence_number: 1
			};
			expect(isErrorEvent(event)).toBe(true);
		});

		it('should return false for failed events (different from error)', () => {
			const event = {
				type: 'response.failed' as const,
				response: { id: 'resp_123' },
				sequence_number: 5
			};
			expect(isErrorEvent(event as any)).toBe(false);
		});
	});
});

// ============================================================================
// Test: Edge Cases
// ============================================================================

describe('edge cases', () => {
	it('should handle rapid successive deltas', async () => {
		const deltas = Array.from({ length: 100 }, (_, i) => ({
			type: 'response.output_text.delta',
			delta: `word${i} `,
			output_index: 0,
			content_index: 0,
			item_id: 'item_1',
			logprobs: [],
			sequence_number: i + 1
		}));

		const chunks = deltas.map((d) => `data: ${JSON.stringify(d)}\n\n`);
		chunks.push('data: [DONE]\n\n');

		const response = createMockResponse(chunks);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(100);
	});

	it('should handle events with special characters in content', async () => {
		const event = {
			type: 'response.output_text.delta',
			delta: 'Code: `const x = "hello\\nworld";`\n\nNewlines and "quotes"',
			output_index: 0,
			content_index: 0,
			item_id: 'item_1',
			logprobs: [],
			sequence_number: 1
		};

		const response = createMockResponse([`data: ${JSON.stringify(event)}\n\n`]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect((events[0] as any).delta).toBe(
			'Code: `const x = "hello\\nworld";`\n\nNewlines and "quotes"'
		);
	});

	it('should handle unicode content', async () => {
		const event = {
			type: 'response.output_text.delta',
			delta: '你好世界 🌍 مرحبا',
			output_index: 0,
			content_index: 0,
			item_id: 'item_1',
			logprobs: [],
			sequence_number: 1
		};

		const response = createMockResponse([`data: ${JSON.stringify(event)}\n\n`]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
		expect((events[0] as any).delta).toBe('你好世界 🌍 مرحبا');
	});

	it('should handle whitespace in data line', async () => {
		const event = { type: 'response.created', response: { id: 'resp_123' }, sequence_number: 0 };

		// Extra spaces after "data:"
		const response = createMockResponse([`data:   ${JSON.stringify(event)}\n\n`]);

		const events = [];
		for await (const e of parseSSEStream(response)) {
			events.push(e);
		}

		expect(events).toHaveLength(1);
	});
});
