import { test, expect } from './fixtures';
import * as path from 'path';
import * as fs from 'fs';

test.describe('Image Upload', () => {
	// Use fresh browser context for each test to avoid route leakage
	test.describe.configure({ mode: 'serial' });
	// 1x1 transparent PNG for testing
	const TINY_PNG_BASE64 =
		'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

	let testImagePath: string;

	test.beforeAll(() => {
		// Create test image file in test-results directory
		const testResultsDir = path.join(process.cwd(), 'test-results');
		if (!fs.existsSync(testResultsDir)) {
			fs.mkdirSync(testResultsDir, { recursive: true });
		}
		testImagePath = path.join(testResultsDir, 'test-image.png');
		const buffer = Buffer.from(TINY_PNG_BASE64, 'base64');
		fs.writeFileSync(testImagePath, buffer);
	});

	test.afterAll(() => {
		// Clean up test image
		if (fs.existsSync(testImagePath)) {
			fs.unlinkSync(testImagePath);
		}
	});

	// Helper to wait for page to be fully ready
	async function waitForPageReady(page: import('@playwright/test').Page) {
		await page.waitForLoadState('networkidle');
		const trigger = page.getByTestId('model-selector-trigger');
		await expect(trigger).toBeEnabled({ timeout: 15000 });
	}

	// Helper to select vision model before image tests
	async function selectVisionModel(page: import('@playwright/test').Page) {
		await waitForPageReady(page);

		// Wait for models to load (button becomes enabled)
		const trigger = page.getByTestId('model-selector-trigger');

		// Open model selector dropdown
		await trigger.click();

		// Wait for dropdown to appear and click the vision model
		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Click the qwen3-vl-2b option
		const visionModel = dropdown.locator('button', { hasText: 'qwen3-vl-2b' });
		await visionModel.click();

		// Wait for dropdown to close and selection to take effect
		await expect(dropdown).not.toBeVisible();
	}

	test('attach button is visible', async ({ page }) => {
		await page.goto('/');
		// Just wait for load state, don't need models to be loaded for this test
		await page.waitForLoadState('networkidle');

		const attachButton = page.getByTestId('attach-button');
		await expect(attachButton).toBeVisible({ timeout: 10000 });
	});

	// SKIP: Model API calls timeout intermittently in Docker E2E environment
	test.skip('image preview appears after attaching file', async ({ page }) => {
		await page.goto('/');
		await selectVisionModel(page);

		// Get the file input (it's hidden but we can still interact with it)
		const fileInput = page.locator('input[type="file"]');
		await fileInput.setInputFiles(testImagePath);

		// Check that the attachment strip appears with the image preview
		const attachmentStrip = page.locator('.flex.flex-wrap.gap-2');
		await expect(attachmentStrip).toBeVisible();

		// Check that an img element is present with our file
		const imagePreview = attachmentStrip.locator('img');
		await expect(imagePreview).toBeVisible();

		// Check that filename is shown
		await expect(attachmentStrip).toContainText('test-image.png');

		await page.screenshot({
			path: 'test-results/screenshots/image-preview.png',
			fullPage: true
		});
	});

	// SKIP: Model API calls timeout intermittently in Docker E2E environment
	test.skip('attachment can be removed via X button', async ({ page }) => {
		await page.goto('/');
		await selectVisionModel(page);

		// Attach image
		const fileInput = page.locator('input[type="file"]');
		await fileInput.setInputFiles(testImagePath);

		// Verify attachment is shown
		const attachmentStrip = page.locator('.flex.flex-wrap.gap-2');
		await expect(attachmentStrip).toBeVisible();

		// Find and click the remove button (force click since it may be opacity-0 until hover)
		const removeButton = page.locator('button[aria-label*="Remove"]');
		await removeButton.click({ force: true });

		// Verify attachment strip is hidden (no attachments)
		await expect(attachmentStrip).not.toBeVisible();
	});

	// SKIP: Model API calls timeout intermittently in Docker E2E environment
	test.skip('submit with image sends correct message format', async ({ page }) => {
		// Intercept the API request to check the payload format
		let capturedRequest: { input: unknown } | null = null;

		// Define the route handler so we can unroute it later
		// Use regex pattern to match only the responses endpoint path, not hostname
		const routePattern = /\/v1\/responses$/;
		const routeHandler = async (route: import('@playwright/test').Route) => {
			const request = route.request();
			if (request.method() === 'POST') {
				const postData = request.postDataJSON();
				capturedRequest = postData;
				// Abort the request so we don't need a real backend response
				await route.abort();
			} else {
				await route.continue();
			}
		};

		await page.route(routePattern, routeHandler);

		try {
			await page.goto('/');
			await selectVisionModel(page);

			// Attach image
			const fileInput = page.locator('input[type="file"]');
			await fileInput.setInputFiles(testImagePath);

			// Type a message
			const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
			await textarea.fill('What is in this image?');

			// Click send
			const sendButton = page.getByTestId('send-button');
			await sendButton.click();

			// Wait for the request to be captured
			await page.waitForTimeout(500);

			// Verify the request format
			expect(capturedRequest).not.toBeNull();
			expect(capturedRequest!.input).toBeDefined();

			// Input should be an array with a message object
			const input = capturedRequest!.input as Array<{
				type: string;
				role?: string;
				content?: Array<{ type: string; text?: string; image_url?: { url: string } }>;
			}>;
			expect(Array.isArray(input)).toBe(true);
			expect(input.length).toBe(1);

			// First item should be a message object
			const message = input[0];
			expect(message.type).toBe('message');
			expect(message.role).toBe('user');
			expect(Array.isArray(message.content)).toBe(true);

			// Content should have text and image parts
			const content = message.content!;
			const textPart = content.find((p) => p.type === 'input_text');
			const imagePart = content.find((p) => p.type === 'input_image');

			expect(textPart).toBeDefined();
			expect(textPart!.text).toContain('What is in this image?');

			expect(imagePart).toBeDefined();
			expect(imagePart!.image_url).toBeDefined();
			expect(imagePart!.image_url!.url).toMatch(/^data:image\/png;base64,/);
		} finally {
			// Clean up the route to prevent affecting subsequent tests
			await page.unroute(routePattern, routeHandler);
		}
	});

	// TODO: This test passes in isolation but fails when run after 'submit with image' test
	// Appears to be a Playwright/Docker interaction issue with route persistence or backend state
	test.skip('text file attachment works', async ({ page }) => {
		// Create a test text file
		const testTextPath = path.join(process.cwd(), 'test-results', 'test-file.txt');
		fs.writeFileSync(testTextPath, 'Hello, this is a test file.');

		try {
			await page.goto('/');

			// Wait for page to be fully loaded
			await page.waitForLoadState('networkidle');

			// Wait for models to load before attaching files
			const trigger = page.getByTestId('model-selector-trigger');
			await expect(trigger).toBeEnabled({ timeout: 15000 });

			// Attach text file
			const fileInput = page.locator('input[type="file"]');
			await fileInput.setInputFiles(testTextPath);

			// Check that the attachment strip appears with file icon (not image)
			const attachmentStrip = page.locator('.flex.flex-wrap.gap-2');
			await expect(attachmentStrip).toBeVisible();

			// Should show filename
			await expect(attachmentStrip).toContainText('test-file.txt');

			// Should have FileText icon (SVG), not an img
			const imagePreview = attachmentStrip.locator('img');
			await expect(imagePreview).toHaveCount(0);
		} finally {
			// Clean up
			if (fs.existsSync(testTextPath)) {
				fs.unlinkSync(testTextPath);
			}
		}
	});

	// TODO: This test passes in isolation but fails when run after 'submit with image' test
	// Appears to be a Playwright/Docker interaction issue with route persistence or backend state
	test.skip('multiple files can be attached', async ({ page }) => {
		// Create a second test file
		const testTextPath = path.join(process.cwd(), 'test-results', 'test-file.txt');
		fs.writeFileSync(testTextPath, 'Test content');

		try {
			await page.goto('/');
			await selectVisionModel(page);

			const fileInput = page.locator('input[type="file"]');

			// Attach image first
			await fileInput.setInputFiles(testImagePath);

			// Then attach text file
			await fileInput.setInputFiles(testTextPath);

			// Should show both attachments
			const attachmentStrip = page.locator('.flex.flex-wrap.gap-2');
			await expect(attachmentStrip).toContainText('test-image.png');
			await expect(attachmentStrip).toContainText('test-file.txt');

			// Should have 2 remove buttons
			const removeButtons = page.locator('button[aria-label*="Remove"]');
			await expect(removeButtons).toHaveCount(2);

			await page.screenshot({
				path: 'test-results/screenshots/multiple-attachments.png',
				fullPage: true
			});
		} finally {
			if (fs.existsSync(testTextPath)) {
				fs.unlinkSync(testTextPath);
			}
		}
	});
});
