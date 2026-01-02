import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/jobs/:id/checkpoints - List checkpoints for a job
export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/jobs/${params.id}/checkpoints`);

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Job not found' });
		}
		throw error(res.status, { message: 'Failed to get checkpoints' });
	}

	return json(await res.json());
};
