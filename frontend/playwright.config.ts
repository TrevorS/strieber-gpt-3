import { defineConfig, devices } from '@playwright/test';

// Always use our own preview server - the Dockerfile.playwright builds with
// VITE_RESPONSES_API_URL baked in so API calls work inside Docker network
export default defineConfig({
	testDir: './e2e',
	outputDir: './test-results',

	// Limit parallel workers to reduce LLM contention
	// Local LLM can't handle many concurrent requests
	workers: process.env.CI ? 1 : 2,

	// Run test files in parallel - each gets own worker with clean state
	fullyParallel: true,

	// Increase global timeout for LLM-dependent tests
	// With test.slow(), effective timeout becomes 240s
	timeout: 120000,

	// Clean up database before running tests
	globalSetup: './e2e/global-setup.ts',

	// Build and run our own preview server with correct API URL
	webServer: {
		command: 'npm run preview',
		port: 4173,
		reuseExistingServer: !process.env.CI
	},

	// Screenshot settings
	use: {
		baseURL: 'http://localhost:4173',
		screenshot: 'on',
		trace: 'on-first-retry'
	},

	// Single browser for now
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] }
		}
	],

	// Reporter
	reporter: [['html', { outputFolder: './playwright-report' }], ['list']]
});
