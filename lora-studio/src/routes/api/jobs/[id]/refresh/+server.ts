import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/jobs/:id/refresh - Refresh job progress from container logs
export const POST: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/jobs/${params.id}/refresh`, {
		method: 'POST'
	});

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Job not found or container not running' });
		}
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to refresh job' });
	}

	return json(await res.json());
};
