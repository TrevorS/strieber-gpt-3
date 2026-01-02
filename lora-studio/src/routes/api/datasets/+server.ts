import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { callTool } from '$lib/server/mcp';
import type { Dataset } from '$lib/server/types';

// GET /api/datasets - List all datasets
export const GET: RequestHandler = async () => {
	try {
		const result = await callTool<{ datasets: Dataset[] }>('lora_list_datasets');
		return json(result.datasets);
	} catch (e) {
		console.error('Failed to list datasets:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to list datasets' });
	}
};

// POST /api/datasets - Create a new dataset
export const POST: RequestHandler = async ({ request }) => {
	try {
		const { name, trigger_token, lora_type } = await request.json();

		if (!name || typeof name !== 'string') {
			throw error(400, { message: 'Dataset name is required' });
		}

		const result = await callTool<Dataset>('lora_create_dataset', {
			name,
			trigger_token: trigger_token || name,
			lora_type: lora_type || 'character'
		});
		return json(result, { status: 201 });
	} catch (e) {
		if (e instanceof Response) throw e;
		console.error('Failed to create dataset:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to create dataset' });
	}
};
