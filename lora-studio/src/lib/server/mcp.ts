// MCP client for lora-trainer
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

const MCP_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

let client: Client | null = null;
let transport: StreamableHTTPClientTransport | null = null;

interface TextContent {
	type: 'text';
	text: string;
}

interface ToolResult {
	content: TextContent[];
}

async function getClient(): Promise<Client> {
	if (client) return client;

	transport = new StreamableHTTPClientTransport(new URL(`${MCP_URL}/mcp`));
	client = new Client({ name: 'lora-studio', version: '1.0.0' });
	await client.connect(transport);

	return client;
}

/**
 * Call an MCP tool with JSON output format.
 * Parses the JSON response from the tool's TextContent.
 */
export async function callTool<T>(tool: string, params: Record<string, unknown> = {}): Promise<T> {
	const mcp = await getClient();

	// Always request JSON format for UI consumption
	const result = (await mcp.callTool({
		name: tool,
		arguments: { ...params, output_format: 'json' }
	})) as ToolResult;

	// MCP returns content array, first item should be TextContent with JSON
	if (!result.content || result.content.length === 0) {
		throw new Error('Empty response from MCP tool');
	}

	const content = result.content[0];
	if (content.type !== 'text') {
		throw new Error(`Unexpected content type: ${content.type}`);
	}

	const data = JSON.parse(content.text);

	// Check for error responses
	if (data.error) {
		throw new Error(data.error);
	}

	return data as T;
}

/**
 * Call an MCP tool that returns image data.
 * Returns the raw image response with base64 data and content type.
 */
export async function callImageTool(
	tool: string,
	params: Record<string, unknown> = {}
): Promise<{ data: string; content_type: string; caption: string | null }> {
	return callTool(tool, params);
}

/**
 * Close the MCP connection (for cleanup).
 */
export async function closeClient(): Promise<void> {
	if (client) {
		await client.close();
		client = null;
		transport = null;
	}
}
