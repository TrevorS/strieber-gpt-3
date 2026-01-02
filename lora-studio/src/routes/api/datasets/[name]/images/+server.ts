import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { callTool } from '$lib/server/mcp';
import type { AddImagesResult } from '$lib/server/types';

// POST /api/datasets/:name/images - Add images to dataset
export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const body = await request.json();

		// Support both single URL and array of URLs
		const sources: string[] = Array.isArray(body.urls) ? body.urls : body.url ? [body.url] : [];

		if (sources.length === 0) {
			throw error(400, { message: 'At least one URL is required' });
		}

		const result = await callTool<AddImagesResult>('lora_add_images', {
			dataset_name: params.name,
			sources
		});

		return json(result);
	} catch (e) {
		if (e instanceof Response) throw e;
		console.error('Failed to add images:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to add images' });
	}
};
