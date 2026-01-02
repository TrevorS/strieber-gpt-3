import { error, json } from '@sveltejs/kit';
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

// DELETE /api/datasets/:name/images/:filename - Delete image from dataset
export const DELETE: RequestHandler = async ({ params, fetch }) => {
	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}/images/${params.filename}`, {
		method: 'DELETE'
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: 'Failed to delete image' }));
		throw error(res.status, { message: err.error || 'Failed to delete image' });
	}

	return new Response(null, { status: 204 });
};

// PUT /api/datasets/:name/images/:filename - Update image caption
export const PUT: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	const res = await fetch(
		`${LORA_API_URL}/api/datasets/${params.name}/images/${params.filename}/caption`,
		{
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ caption: body.caption })
		}
	);

	if (!res.ok) {
		const err = await res.json().catch(() => ({ error: 'Failed to update caption' }));
		throw error(res.status, { message: err.error || 'Failed to update caption' });
	}

	return json(await res.json());
};
