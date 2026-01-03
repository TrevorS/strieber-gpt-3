import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/loras - List all LoRAs
export const GET: RequestHandler = async ({ fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/loras`);
	if (!res.ok) {
		throw error(res.status, { message: 'Failed to list LoRAs' });
	}
	return json(await res.json());
};
