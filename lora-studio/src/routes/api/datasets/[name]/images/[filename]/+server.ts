import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { callImageTool } from '$lib/server/mcp';

// GET /api/datasets/:name/images/:filename - Serve dataset image via MCP
export const GET: RequestHandler = async ({ params }) => {
	const { name, filename } = params;

	try {
		const result = await callImageTool('lora_get_image', {
			dataset_name: name,
			filename
		});

		// Decode base64 to binary
		const binary = Uint8Array.from(atob(result.data), (c) => c.charCodeAt(0));

		return new Response(binary, {
			headers: {
				'Content-Type': result.content_type,
				'Cache-Control': 'public, max-age=3600'
			}
		});
	} catch (e) {
		console.error('Failed to get image:', e);
		throw error(404, { message: 'Image not found' });
	}
};
