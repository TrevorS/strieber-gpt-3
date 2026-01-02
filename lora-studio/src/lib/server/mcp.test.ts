import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockCallTool = vi.fn();
const mockConnect = vi.fn();
const mockClose = vi.fn();

// Mock the MCP SDK before any imports
vi.mock('@modelcontextprotocol/sdk/client/index.js', () => ({
	Client: class MockClient {
		connect = mockConnect;
		callTool = mockCallTool;
		close = mockClose;
	}
}));

vi.mock('@modelcontextprotocol/sdk/client/streamableHttp.js', () => ({
	StreamableHTTPClientTransport: class MockTransport {
		constructor() {}
	}
}));

describe('callTool', () => {
	beforeEach(async () => {
		vi.clearAllMocks();
		// Reset module to clear cached client
		vi.resetModules();
	});

	it('should call MCP tool with output_format: json', async () => {
		mockCallTool.mockResolvedValueOnce({
			content: [{ type: 'text', text: '{"datasets": []}' }]
		});

		const { callTool } = await import('./mcp');
		await callTool('lora_list_datasets');

		expect(mockCallTool).toHaveBeenCalledWith({
			name: 'lora_list_datasets',
			arguments: { output_format: 'json' }
		});
	});

	it('should merge params with output_format', async () => {
		mockCallTool.mockResolvedValueOnce({
			content: [{ type: 'text', text: '{"name": "test"}' }]
		});

		const { callTool } = await import('./mcp');
		await callTool('lora_get_dataset', { name: 'test' });

		expect(mockCallTool).toHaveBeenCalledWith({
			name: 'lora_get_dataset',
			arguments: { name: 'test', output_format: 'json' }
		});
	});

	it('should parse JSON response', async () => {
		const expectedData = { datasets: [{ name: 'ds1' }, { name: 'ds2' }] };

		mockCallTool.mockResolvedValueOnce({
			content: [{ type: 'text', text: JSON.stringify(expectedData) }]
		});

		const { callTool } = await import('./mcp');
		const result = await callTool<typeof expectedData>('lora_list_datasets');
		expect(result).toEqual(expectedData);
	});

	it('should throw on error response', async () => {
		mockCallTool.mockResolvedValueOnce({
			content: [{ type: 'text', text: '{"error": "Dataset not found"}' }]
		});

		const { callTool } = await import('./mcp');
		await expect(callTool('lora_get_dataset', { name: 'missing' })).rejects.toThrow(
			'Dataset not found'
		);
	});

	it('should throw on empty response', async () => {
		mockCallTool.mockResolvedValueOnce({
			content: []
		});

		const { callTool } = await import('./mcp');
		await expect(callTool('lora_list_datasets')).rejects.toThrow('Empty response from MCP tool');
	});
});
