import { describe, expect, it, vi } from 'vitest';
import { DELETE, GET } from '../+server';
import type { RequestEvent } from '@sveltejs/kit';

type MockEvent = RequestEvent<{ name: string }, '/api/datasets/[name]'>;

const mockDatasetDetail = {
	name: 'test_dataset',
	trigger_token: 'tst',
	lora_type: 'character',
	description: 'Test dataset',
	image_count: 3,
	has_captions: true,
	created_at: '2025-01-01T00:00:00Z',
	images: [
		{ filename: '001.png', caption: 'test caption' },
		{ filename: '002.png', caption: null }
	]
};

function createMockEvent(
	params: Record<string, string>,
	overrides: Record<string, unknown> = {}
): MockEvent {
	return {
		fetch: vi.fn(),
		request: new Request('http://localhost/api/datasets/test'),
		params,
		url: new URL('http://localhost/api/datasets/test'),
		locals: {},
		platform: undefined,
		route: { id: '/api/datasets/[name]' },
		cookies: {},
		getClientAddress: () => '127.0.0.1',
		isDataRequest: false,
		isSubRequest: false,
		setHeaders: vi.fn(),
		...overrides
	} as unknown as MockEvent;
}

describe('GET /api/datasets/:name', () => {
	it('should return dataset details from backend', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(mockDatasetDetail)
		});

		const event = createMockEvent({ name: 'test_dataset' }, { fetch: mockFetch });
		const response = await GET(event);
		const data = await response.json();

		expect(mockFetch).toHaveBeenCalledWith(
			'http://mcp-lora-trainer:8000/api/datasets/test_dataset'
		);
		expect(data.name).toBe('test_dataset');
		expect(data.images).toHaveLength(2);
	});

	it('should throw 404 when dataset not found', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: false,
			status: 404
		});

		const event = createMockEvent({ name: 'nonexistent' }, { fetch: mockFetch });

		await expect(GET(event)).rejects.toMatchObject({
			status: 404,
			body: { message: 'Dataset not found' }
		});
	});
});

describe('DELETE /api/datasets/:name', () => {
	it('should delete dataset and return 204', async () => {
		const mockFetch = vi.fn().mockResolvedValue({ ok: true });

		const event = createMockEvent({ name: 'test_dataset' }, { fetch: mockFetch });
		const response = await DELETE(event);

		expect(mockFetch).toHaveBeenCalledWith(
			'http://mcp-lora-trainer:8000/api/datasets/test_dataset',
			{ method: 'DELETE' }
		);
		expect(response.status).toBe(204);
	});

	it('should throw error on delete failure', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: false,
			status: 500
		});

		const event = createMockEvent({ name: 'test_dataset' }, { fetch: mockFetch });

		await expect(DELETE(event)).rejects.toMatchObject({
			status: 500,
			body: { message: 'Failed to delete dataset' }
		});
	});
});
