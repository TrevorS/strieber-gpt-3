import { test, expect } from '@playwright/test';

test.describe('Chat Functionality', () => {
	// Increase timeout for tests that wait for LLM responses
	test.setTimeout(60000);

	test('sends message and receives streaming response', async ({ page }) => {
		await page.goto('/');

		// Find input and send button
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Type a simple message
		await textarea.fill('Say "Hello World" and nothing else.');
		await sendButton.click();

		// Verify user message appears (use div to exclude button)
		const userMessage = page.locator('div.bg-primary').first();
		await expect(userMessage).toBeVisible();
		await expect(userMessage).toContainText('Say "Hello World"');

		// Wait for assistant response (streaming completes)
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });

		// Verify response has content
		await expect(assistantMessage).not.toBeEmpty();

		// Screenshot for verification
		await page.screenshot({
			path: 'test-results/screenshots/chat-basic.png',
			fullPage: true
		});
	});

	test('renders markdown with code blocks and syntax highlighting', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Request a simple code example
		await textarea.fill('Show me a Python hello world in a code block. Just the code, nothing else.');
		await sendButton.click();

		// Wait for response with content
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });
		await expect(assistantMessage).not.toBeEmpty({ timeout: 30000 });

		// Small wait for rendering
		await page.waitForTimeout(500);

		// Find code block - may or may not exist depending on LLM response
		const codeBlock = assistantMessage.locator('pre code');
		const hasCodeBlock = (await codeBlock.count()) > 0;

		if (hasCodeBlock) {
			await expect(codeBlock).toBeVisible();

			// Verify syntax highlighting is applied (hljs class or language- class)
			const hasHighlighting = await codeBlock.evaluate((el) => {
				return el.classList.contains('hljs') || el.className.includes('language-');
			});
			expect(hasHighlighting).toBe(true);

			// Verify some syntax tokens exist (spans with hljs- classes)
			const hasSyntaxTokens = await codeBlock.evaluate((el) => {
				return el.querySelectorAll('span[class*="hljs-"]').length > 0;
			});
			expect(hasSyntaxTokens).toBe(true);
		}

		// Verify response has some content regardless
		const responseText = await assistantMessage.textContent();
		expect(responseText?.length).toBeGreaterThan(0);

		await page.screenshot({
			path: 'test-results/screenshots/code-block-highlighting.png',
			fullPage: true
		});
	});

	test('multi-turn conversation flow works', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// First message
		await textarea.fill('Say "one" only.');
		await sendButton.click();

		// Wait for first response with content (streaming complete)
		const firstResponse = page.locator('.bg-muted').first();
		await expect(firstResponse).toBeVisible({ timeout: 30000 });
		await expect(firstResponse).not.toBeEmpty({ timeout: 30000 });

		// Verify first user message visible
		const firstUserMessage = page.locator('div.bg-primary').first();
		await expect(firstUserMessage).toContainText('one');

		// Small wait for any state to settle
		await page.waitForTimeout(500);

		// Second message
		await textarea.fill('Say "two" only.');
		await sendButton.click();

		// Wait for second response with content
		const secondResponse = page.locator('.bg-muted').nth(1);
		await expect(secondResponse).toBeVisible({ timeout: 30000 });
		await expect(secondResponse).not.toBeEmpty({ timeout: 30000 });

		// Verify we have 2 user messages and 2 assistant responses
		const userMessages = page.locator('div.bg-primary');
		const assistantMessages = page.locator('.bg-muted');
		await expect(userMessages).toHaveCount(2);
		await expect(assistantMessages).toHaveCount(2);

		await page.screenshot({
			path: 'test-results/screenshots/chat-multi-turn.png',
			fullPage: true
		});
	});

	test('input states work correctly', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Send button should be disabled when empty
		await expect(sendButton).toBeDisabled();

		// Type something - button should enable
		await textarea.fill('Test message');
		await expect(sendButton).toBeEnabled();

		// Clear - button should disable again
		await textarea.fill('');
		await expect(sendButton).toBeDisabled();

		// Send a message
		await textarea.fill('Say "ok" and nothing else.');
		await sendButton.click();

		// Button should be disabled during streaming (textarea stays enabled for typing)
		await expect(sendButton).toBeDisabled({ timeout: 5000 });

		// Wait for streaming to complete
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });

		// Button should be re-enabled after response (with empty input, button stays disabled)
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		await page.screenshot({
			path: 'test-results/screenshots/chat-input-states.png',
			fullPage: true
		});
	});

	test('enter key sends message', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');

		// Click to focus, type message, then press Enter
		await textarea.click();
		await textarea.fill('Say "hello" only.');
		await textarea.press('Enter');

		// Verify user message appears (use div to exclude button)
		const userMessage = page.locator('div.bg-primary').first();
		await expect(userMessage).toBeVisible();
		await expect(userMessage).toContainText('Say "hello"');

		// Wait for response
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });
	});

	test('shift+enter adds newline instead of sending', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');

		// Type, press Shift+Enter, type more
		await textarea.fill('Line 1');
		await textarea.press('Shift+Enter');
		await textarea.type('Line 2');

		// Verify textarea has multiline content
		const value = await textarea.inputValue();
		expect(value).toContain('Line 1');
		expect(value).toContain('Line 2');

		// No message should be sent yet (use div to target messages, not button)
		const userMessage = page.locator('div.bg-primary');
		await expect(userMessage).toHaveCount(0);
	});
});
