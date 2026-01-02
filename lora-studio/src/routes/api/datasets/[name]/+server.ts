import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { callTool } from '$lib/server/mcp';
import type { DatasetInfo } from '$lib/server/types';

// GET /api/datasets/:name - Get dataset details with images
export const GET: RequestHandler = async ({ params }) => {
	try {
		const result = await callTool<DatasetInfo>('lora_get_dataset', {
			name: params.name
		});
		return json(result);
	} catch (e) {
		console.error('Failed to get dataset:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to get dataset' });
	}
};

// DELETE /api/datasets/:name - Delete a dataset
export const DELETE: RequestHandler = async ({ params }) => {
	try {
		await callTool('lora_delete_dataset', { name: params.name });
		return new Response(null, { status: 204 });
	} catch (e) {
		console.error('Failed to delete dataset:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to delete dataset' });
	}
};
