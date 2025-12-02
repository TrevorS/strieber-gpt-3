/**
 * OpenAI API Client Wrapper
 *
 * Creates an OpenAI client configured for the local Responses API backend.
 * Uses environment variable VITE_RESPONSES_API_URL for the base URL.
 */
import OpenAI from 'openai';

/**
 * Get the default API URL based on the current browser location.
 * This allows the app to work when accessed via IP or hostname.
 */
function getDefaultApiUrl(): string {
	if (typeof window !== 'undefined') {
		// In browser: use same host but port 9150
		const { protocol, hostname } = window.location;
		return `${protocol}//${hostname}:9150/v1`;
	}
	// SSR fallback
	return 'http://localhost:9150/v1';
}

/**
 * Creates a new OpenAI client instance.
 *
 * @param baseURL - Override the base URL (defaults to VITE_RESPONSES_API_URL env var or auto-detected)
 * @returns Configured OpenAI client
 */
export function createClient(baseURL?: string): OpenAI {
	return new OpenAI({
		baseURL: baseURL ?? import.meta.env.VITE_RESPONSES_API_URL ?? getDefaultApiUrl(),
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
	return import.meta.env.VITE_RESPONSES_API_URL ?? getDefaultApiUrl();
}
