import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/loras/[name]/test - Generate test image
export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	const res = await fetch(`${LORA_API_URL}/api/loras/${params.name}/test`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw error(res.status, { message: err.error || 'Failed to generate test image' });
	}

	return json(await res.json());
};
