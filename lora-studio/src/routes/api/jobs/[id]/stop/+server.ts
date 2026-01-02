import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/jobs/:id/stop - Stop a running job
export const POST: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/jobs/${params.id}/stop`, {
		method: 'POST'
	});

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Job not found' });
		}
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to stop job' });
	}

	return json(await res.json());
};
