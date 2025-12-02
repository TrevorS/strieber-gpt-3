import { defineConfig, devices } from '@playwright/test';

// Always use our own preview server - the Dockerfile.playwright builds with
// VITE_RESPONSES_API_URL baked in so API calls work inside Docker network
export default defineConfig({
	testDir: './e2e',
	outputDir: './test-results',

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
