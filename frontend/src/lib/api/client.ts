/**
 * OpenAI API Client Wrapper
 *
 * Creates an OpenAI client configured for the local Responses API backend.
 * Uses environment variable VITE_RESPONSES_API_URL for the base URL.
 */
import OpenAI from 'openai';

/**
 * Default Responses API URL (local development)
 */
const DEFAULT_API_URL = 'http://localhost:9150/v1';

/**
 * Creates a new OpenAI client instance.
 *
 * @param baseURL - Override the base URL (defaults to VITE_RESPONSES_API_URL env var or localhost:9150)
 * @returns Configured OpenAI client
 */
export function createClient(baseURL?: string): OpenAI {
	return new OpenAI({
		baseURL: baseURL ?? import.meta.env.VITE_RESPONSES_API_URL ?? DEFAULT_API_URL,
		apiKey: 'not-needed', // Local backend doesn't require auth
		dangerouslyAllowBrowser: true // Required for browser context
	});
}

/**
 * Default singleton client instance for convenience.
 * Import this for standard usage throughout the app.
 */
export const client = createClient();

/**
 * Get the configured API base URL.
 * Useful for direct fetch calls (e.g., streaming).
 */
export function getApiBaseUrl(): string {
	return import.meta.env.VITE_RESPONSES_API_URL ?? DEFAULT_API_URL;
}
