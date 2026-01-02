import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/jobs/:id/promote - Promote a checkpoint to LoRAs directory
export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	if (!body.checkpoint_name || typeof body.checkpoint_name !== 'string') {
		throw error(400, { message: 'checkpoint_name is required' });
	}

	const res = await fetch(`${LORA_API_URL}/api/jobs/${params.id}/promote`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Job or checkpoint not found' });
		}
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to promote checkpoint' });
	}

	return json(await res.json());
};
