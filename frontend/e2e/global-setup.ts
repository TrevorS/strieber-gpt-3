import { FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
	// Clear all test conversations before test run
	// In Docker: use internal network URL (responses-api:8000)
	// Locally: use localhost:9150
	// The VITE_ prefix is for browser builds; we detect Docker via PWD
	const isDocker = process.cwd() === '/app';
	const apiUrl = isDocker
		? 'http://responses-api:8000/v1'
		: (process.env.RESPONSES_API_URL || 'http://localhost:9150/v1');

	console.log(`[Global Setup] Running in ${isDocker ? 'Docker' : 'local'} mode, API: ${apiUrl}`);

	try {
		let totalDeleted = 0;
		let hasMore = true;

		// Loop through all conversations using pagination
		while (hasMore) {
			// Use limit=100 which is the max allowed by the API
			const listResponse = await fetch(`${apiUrl}/conversations?limit=100&order=desc`);
			if (!listResponse.ok) {
				console.warn(`[Global Setup] Could not fetch conversations: ${listResponse.status}`);
				return;
			}

			const result = await listResponse.json();
			const conversations = result.data || [];
			hasMore = result.has_more || false;

			if (conversations.length === 0) {
				break;
			}

			console.log(`[Global Setup] Found ${conversations.length} conversations to delete (has_more: ${hasMore})`);

			// Delete each conversation
			for (const conv of conversations) {
				const deleteResponse = await fetch(`${apiUrl}/conversations/${conv.id}`, {
					method: 'DELETE'
				});
				if (deleteResponse.ok) {
					totalDeleted++;
				}
			}
		}

		if (totalDeleted > 0) {
			console.log(`[Global Setup] Deleted ${totalDeleted} test conversations`);
		} else {
			console.log('[Global Setup] No conversations to clean up');
		}
	} catch (error) {
		// Don't fail tests if cleanup fails - just warn
		console.warn('[Global Setup] Could not clean database:', error);
	}
}

export default globalSetup;
