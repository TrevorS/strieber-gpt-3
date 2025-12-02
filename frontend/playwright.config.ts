import { defineConfig, devices } from '@playwright/test';

// In CI/Docker, use the chat-ui service; locally, use preview server
const useExternalServer = process.env.CI === 'true';
const baseURL = useExternalServer ? 'http://chat-ui:3000' : 'http://localhost:4173';

export default defineConfig({
	testDir: './e2e',
	outputDir: './test-results',

	// In CI, use external chat-ui service; locally, build and run preview
	webServer: useExternalServer
		? undefined
		: {
				command: 'npm run build && npm run preview',
				port: 4173,
				reuseExistingServer: true
			},

	// Screenshot settings
	use: {
		baseURL,
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
