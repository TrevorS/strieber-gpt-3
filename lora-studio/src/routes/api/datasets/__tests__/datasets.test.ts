import { describe, it, expect, vi } from 'vitest';
import { GET, POST } from '../+server';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MockEvent = any;

// Mock dataset response
const mockDatasets = [
	{
		name: 'test_dataset',
		trigger_token: 'tst',
		lora_type: 'character',
		description: null,
		image_count: 5,
		has_captions: true,
		created_at: '2025-01-01T00:00:00Z'
	}
];

function createMockEvent(overrides: Record<string, unknown> = {}): MockEvent {
	return {
		fetch: vi.fn(),
		request: new Request('http://localhost/api/datasets'),
		params: {},
		url: new URL('http://localhost/api/datasets'),
		locals: {},
		platform: undefined,
		route: { id: '/api/datasets' },
		cookies: {},
		getClientAddress: () => '127.0.0.1',
		isDataRequest: false,
		isSubRequest: false,
		setHeaders: vi.fn(),
		...overrides
	};
}

describe('GET /api/datasets', () => {
	it('should return datasets from backend', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(mockDatasets)
		});

		const event = createMockEvent({ fetch: mockFetch });
		const response = await GET(event);
		const data = await response.json();

		expect(mockFetch).toHaveBeenCalledWith('http://mcp-lora-trainer:8000/api/datasets');
		expect(data).toEqual(mockDatasets);
	});

	it('should throw error on backend failure', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: false,
			status: 500
		});

		const event = createMockEvent({ fetch: mockFetch });

		await expect(GET(event)).rejects.toMatchObject({
			status: 500,
			body: { message: 'Failed to list datasets' }
		});
	});
});

describe('POST /api/datasets', () => {
	it('should create dataset with valid input', async () => {
		const createdDataset = { ...mockDatasets[0], name: 'new_dataset' };
		const mockFetch = vi.fn().mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(createdDataset)
		});

		const event = createMockEvent({
			fetch: mockFetch,
			request: new Request('http://localhost/api/datasets', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name: 'new_dataset', trigger_token: 'nds' })
			})
		});

		const response = await POST(event);
		const data = await response.json();

		expect(response.status).toBe(201);
		expect(data.name).toBe('new_dataset');
		expect(mockFetch).toHaveBeenCalledWith(
			'http://mcp-lora-trainer:8000/api/datasets',
			expect.objectContaining({
				method: 'POST',
				headers: { 'Content-Type': 'application/json' }
			})
		);
	});

	it('should use name as trigger_token if not provided', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: true,
			json: () => Promise.resolve(mockDatasets[0])
		});

		const event = createMockEvent({
			fetch: mockFetch,
			request: new Request('http://localhost/api/datasets', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name: 'my_dataset' })
			})
		});

		await POST(event);

		const fetchCall = mockFetch.mock.calls[0];
		const body = JSON.parse(fetchCall[1].body);
		expect(body.trigger_token).toBe('my_dataset');
		expect(body.lora_type).toBe('character');
	});

	it('should reject request without name', async () => {
		const event = createMockEvent({
			request: new Request('http://localhost/api/datasets', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ trigger_token: 'abc' })
			})
		});

		await expect(POST(event)).rejects.toMatchObject({
			status: 400,
			body: { message: 'Dataset name is required' }
		});
	});

	it('should forward backend error messages', async () => {
		const mockFetch = vi.fn().mockResolvedValue({
			ok: false,
			status: 400,
			json: () => Promise.resolve({ error: 'Dataset already exists' })
		});

		const event = createMockEvent({
			fetch: mockFetch,
			request: new Request('http://localhost/api/datasets', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name: 'existing_dataset' })
			})
		});

		await expect(POST(event)).rejects.toMatchObject({
			status: 400,
			body: { message: 'Dataset already exists' }
		});
	});
});
