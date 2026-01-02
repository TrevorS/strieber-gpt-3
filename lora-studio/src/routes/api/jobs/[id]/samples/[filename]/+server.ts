import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// GET /api/jobs/:id/samples/:filename - Get sample image
export const GET: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/jobs/${params.id}/samples/${params.filename}`);

	if (!res.ok) {
		if (res.status === 404) {
			throw error(404, { message: 'Sample not found' });
		}
		throw error(res.status, { message: 'Failed to get sample' });
	}

	const contentType = res.headers.get('content-type') || 'image/jpeg';
	const imageData = await res.arrayBuffer();

	return new Response(imageData, {
		headers: {
			'Content-Type': contentType,
			'Cache-Control': 'public, max-age=3600'
		}
	});
};
