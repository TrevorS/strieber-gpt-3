import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/datasets/:name/caption - Auto-caption images
export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}/caption`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			style: body.style || 'detailed',
			overwrite: body.overwrite ?? false,
			image_name: body.image_name
		})
	});

	if (!res.ok) {
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to caption images' });
	}

	return json(await res.json());
};
