import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/datasets/:name - Get dataset details
export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}`);
	if (!res.ok) {
		throw error(res.status, { message: 'Dataset not found' });
	}
	return json(await res.json());
};

// DELETE /api/datasets/:name - Delete dataset
export const DELETE: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}`, {
		method: 'DELETE'
	});
	if (!res.ok) {
		throw error(res.status, { message: 'Failed to delete dataset' });
	}
	return new Response(null, { status: 204 });
};
