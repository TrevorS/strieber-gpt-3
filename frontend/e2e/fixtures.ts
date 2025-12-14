import { test as base } from '@playwright/test';

// Extend base test with cleanup fixture that runs before each worker
export const test = base.extend({
	// Auto-fixture: runs before each test file (worker scope)
	cleanupConversations: [
		async ({}, use) => {
			// Determine API URL based on environment
			const isDocker = process.cwd() === '/app';
			const apiUrl = isDocker
				? 'http://responses-api:8000/v1'
				: process.env.RESPONSES_API_URL || 'http://localhost:9150/v1';

			try {
				// Fetch all conversations
				const resp = await fetch(`${apiUrl}/conversations?limit=100`);
				if (resp.ok) {
					const { data } = await resp.json();
					// Delete each conversation
					for (const conv of data || []) {
						await fetch(`${apiUrl}/conversations/${conv.id}`, {
							method: 'DELETE'
						});
					}
					console.log(`[Fixture] Cleaned ${data?.length || 0} conversations before worker`);
				}
			} catch (e) {
				// Ignore errors - cleanup is best-effort
				console.log('[Fixture] Cleanup failed (non-fatal):', e);
			}

			await use();
		},
		{ auto: true, scope: 'worker' }
	]
});

export { expect } from '@playwright/test';
