import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/datasets - List all datasets
export const GET: RequestHandler = async ({ fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/datasets`);
	if (!res.ok) {
		throw error(res.status, { message: 'Failed to list datasets' });
	}
	return json(await res.json());
};

// POST /api/datasets - Create a new dataset
export const POST: RequestHandler = async ({ request, fetch }) => {
	const body = await request.json();

	if (!body.name || typeof body.name !== 'string') {
		throw error(400, { message: 'Dataset name is required' });
	}

	const res = await fetch(`${LORA_API_URL}/api/datasets`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			name: body.name,
			trigger_token: body.trigger_token || body.name,
			lora_type: body.lora_type || 'character'
		})
	});

	if (!res.ok) {
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to create dataset' });
	}

	return json(await res.json(), { status: 201 });
};
