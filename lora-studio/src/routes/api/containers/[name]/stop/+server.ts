import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/containers/:name/stop - Stop a container
export const POST: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/containers/${params.name}/stop`, {
		method: 'POST'
	});

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Container not found' });
		}
		if (res.status === 403) {
			throw error(403, { message: 'Cannot stop this container' });
		}
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to stop container' });
	}

	return json(await res.json());
};
