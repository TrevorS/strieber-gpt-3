import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	testDir: './e2e',
	outputDir: './test-results',

	// Run production build
	webServer: {
		command: 'npm run build && npm run preview',
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
