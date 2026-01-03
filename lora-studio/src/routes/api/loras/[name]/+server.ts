import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/loras/[name] - Get LoRA details
export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/loras/${params.name}`);
	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw error(res.status, { message: err.error || 'Failed to get LoRA' });
	}
	return json(await res.json());
};

// PUT /api/loras/[name] - Rename LoRA
export const PUT: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	const res = await fetch(`${LORA_API_URL}/api/loras/${params.name}`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw error(res.status, { message: err.error || 'Failed to rename LoRA' });
	}

	return json(await res.json());
};

// DELETE /api/loras/[name] - Delete LoRA
export const DELETE: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/loras/${params.name}`, {
		method: 'DELETE'
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({}));
		throw error(res.status, { message: err.error || 'Failed to delete LoRA' });
	}

	return new Response(null, { status: 204 });
};
