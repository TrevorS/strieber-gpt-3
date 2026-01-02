import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/containers - List all containers
export const GET: RequestHandler = async ({ fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/containers`);

	if (!res.ok) {
		const err = await res.json();
		return json({ error: err.error || 'Failed to list containers' }, { status: res.status });
	}

	return json(await res.json());
};
