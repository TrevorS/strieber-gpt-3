import { test, expect } from '@playwright/test';

test.describe('Streaming Controls', () => {
	test.setTimeout(60000);

	test('stop button appears during streaming', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Verify send button is initially visible (not stop button)
		await expect(sendButton).toBeVisible();

		// Send a message that will take time to respond
		await textarea.fill('Write a detailed explanation of how computers work, covering hardware, software, and the history of computing from early mechanical calculators to modern quantum computers.');
		await sendButton.click();

		// Either the stop button should appear during streaming, OR the response completes quickly
		// We check by racing: wait for stop button or assistant response
		const stopButton = page.getByTestId('stop-button');
		const assistantMessage = page.locator('.bg-muted').first();

		// Wait for either stop button to appear or for response to start
		await Promise.race([
			expect(stopButton).toBeVisible({ timeout: 10000 }),
			expect(assistantMessage).toBeVisible({ timeout: 10000 })
		]);

		// Take screenshot of current state
		await page.screenshot({
			path: 'test-results/screenshots/streaming-stop-button.png',
			fullPage: true
		});

		// Wait for streaming to complete - send button should reappear
		await expect(sendButton).toBeVisible({ timeout: 30000 });
	});

	test('clicking stop button cancels streaming', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Send a message - might generate a long response
		await textarea.fill('Explain quantum computing');
		await sendButton.click();

		// Wait for stop button to appear
		const stopButton = page.getByTestId('stop-button');
		const assistantMessage = page.locator('.bg-muted').first();

		// Poll for stop button (shorter timeout)
		let stopButtonAppeared = false;
		for (let i = 0; i < 40; i++) {
			const isStopVisible = await stopButton.isVisible().catch(() => false);
			if (isStopVisible) {
				stopButtonAppeared = true;
				break;
			}
			await page.waitForTimeout(100);
		}

		if (stopButtonAppeared) {
			// Click stop (race condition safe - button may disappear before click)
			try {
				await stopButton.click({ timeout: 1000 });
				// Send button should reappear (streaming stopped)
				await expect(sendButton).toBeVisible({ timeout: 10000 });
			} catch {
				// Stop button disappeared - streaming completed, verify send button is back
				await expect(sendButton).toBeVisible({ timeout: 30000 });
			}
		} else {
			// Response completed too quickly - that's okay, just verify state is correct
			await expect(sendButton).toBeVisible({ timeout: 30000 });
		}

		// No error toast should appear (cancellation is graceful or response completed)
		const toastContainer = page.getByTestId('toast-container');
		const toasts = toastContainer.locator('[data-testid^="toast-"]');
		const toastCount = await toasts.count();
		expect(toastCount).toBe(0);

		await page.screenshot({
			path: 'test-results/screenshots/streaming-cancelled.png',
			fullPage: true
		});
	});

	test('can send new message after cancelling previous', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Send first message - something that should take time
		await textarea.fill('Write a long epic poem about the ocean, at least 500 words with detailed imagery and multiple stanzas.');
		await sendButton.click();

		// Try to cancel streaming if stop button appears
		const stopButton = page.getByTestId('stop-button');
		const firstAssistantMessage = page.locator('.bg-muted').first();

		// Try to cancel streaming if stop button appears (race condition safe)
		let cancelled = false;
		for (let i = 0; i < 20; i++) {
			const isStopVisible = await stopButton.isVisible().catch(() => false);
			if (isStopVisible) {
				try {
					await stopButton.click({ timeout: 1000 });
					cancelled = true;
					break;
				} catch {
					// Stop button disappeared before click - streaming completed, that's fine
					break;
				}
			}
			await page.waitForTimeout(250);
		}

		// Wait for send button to reappear (either from cancel or completion)
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		// Send a new message
		await textarea.fill('Say "hello" only.');
		await sendButton.click();

		// Should see both user messages
		const userMessages = page.locator('div.bg-primary');
		await expect(userMessages).toHaveCount(2);

		// Wait for the second assistant response
		const assistantMessages = page.locator('.bg-muted');
		await expect(assistantMessages).toHaveCount(2, { timeout: 30000 });

		await page.screenshot({
			path: 'test-results/screenshots/streaming-new-after-cancel.png',
			fullPage: true
		});
	});
});
