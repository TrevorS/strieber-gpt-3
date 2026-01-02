import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { callTool } from '$lib/server/mcp';
import type { CaptionResult } from '$lib/server/types';

// POST /api/datasets/:name/caption - Auto-caption images
export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const body = await request.json();

		const result = await callTool<CaptionResult>('lora_caption', {
			dataset_name: params.name,
			style: body.style || 'tags', // tags, natural, booru
			overwrite: body.overwrite ?? false
		});

		return json(result);
	} catch (e) {
		console.error('Failed to caption images:', e);
		throw error(500, { message: e instanceof Error ? e.message : 'Failed to caption images' });
	}
};
