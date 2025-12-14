/**
 * Models API Client
 *
 * Fetches available models from the backend /v1/models endpoint.
 */

import { getApiBaseUrl } from './client';
import { logger } from '$lib/utils/logger';

export interface Model {
	id: string;
	object?: string;
	created?: number;
	owned_by?: string;
	supports_vision?: boolean;
	/** Which tools this model supports. null = all tools, [] = no tools */
	supported_tools?: string[] | null;
	/** Model capabilities (e.g., 'task', 'vision', 'reasoning') */
	capabilities?: string[];
}

export interface ModelsResponse {
	object: string;
	data: Model[];
}

/**
 * Fetch available models from the API.
 * Returns an empty array on error to gracefully degrade.
 */
export async function fetchModels(): Promise<Model[]> {
	const requestId = Math.random().toString(16).slice(2, 10);

	try {
		logger.debug('api', 'Fetching models', { requestId });

		const response = await fetch(`${getApiBaseUrl()}/models`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json'
			}
		});

		if (!response.ok) {
			throw new Error(`API error ${response.status}: ${response.statusText}`);
		}

		const data: ModelsResponse = await response.json();
		logger.debug('api', 'Models fetched', { requestId, count: data.data.length });

		return data.data;
	} catch (error) {
		logger.error('api', 'Failed to fetch models', {
			requestId,
			error: error instanceof Error ? error.message : String(error)
		});
		return [];
	}
}
