import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/jobs - List all training jobs
export const GET: RequestHandler = async ({ url, fetch }) => {
	const status = url.searchParams.get('status');
	const queryString = status ? `?status=${status}` : '';

	const res = await fetch(`${LORA_API_URL}/api/jobs${queryString}`);
	if (!res.ok) {
		throw error(res.status, { message: 'Failed to list jobs' });
	}
	return json(await res.json());
};

// POST /api/jobs - Start a new training job
export const POST: RequestHandler = async ({ request, fetch }) => {
	const body = await request.json();

	if (!body.dataset_name || typeof body.dataset_name !== 'string') {
		throw error(400, { message: 'dataset_name is required' });
	}

	const res = await fetch(`${LORA_API_URL}/api/jobs`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});

	if (!res.ok) {
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to start training' });
	}

	return json(await res.json(), { status: 201 });
};
