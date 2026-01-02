import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const LORA_API_URL = process.env.LORA_TRAINER_URL || 'http://mcp-lora-trainer:8000';

// POST /api/datasets/:name/images - Add images from URLs
export const POST: RequestHandler = async ({ params, request, fetch }) => {
	const body = await request.json();

	// Support both single URL and array of URLs
	const urls: string[] = Array.isArray(body.urls) ? body.urls : body.url ? [body.url] : [];

	if (urls.length === 0) {
		throw error(400, { message: 'At least one URL is required' });
	}

	const res = await fetch(`${LORA_API_URL}/api/datasets/${params.name}/images`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			urls,
			auto_caption: body.auto_caption ?? false,
			caption_style: body.caption_style || 'detailed',
			preprocess: body.preprocess ?? true,
			crop_mode: body.crop_mode || 'smart'
		})
	});

	if (!res.ok) {
		const err = await res.json();
		throw error(res.status, { message: err.error || 'Failed to add images' });
	}

	return json(await res.json());
};
