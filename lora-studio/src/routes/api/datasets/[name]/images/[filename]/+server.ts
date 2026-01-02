import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/datasets/:name/images/:filename - Proxy image from backend
export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}/images/${params.filename}`);

	if (!res.ok) {
		throw error(404, { message: 'Image not found' });
	}

	return new Response(res.body, {
		headers: {
			'Content-Type': res.headers.get('Content-Type') || 'image/png',
			'Cache-Control': 'public, max-age=3600'
		}
	});
};
