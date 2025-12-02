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

		// Request a Python code example with function definition
		await textarea.fill(
			'Write a Python function called greet that takes a name parameter and prints a greeting. Include the function definition and call it. Use a code block with python specified.'
		);
		await sendButton.click();

		// Wait for response
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });

		// Wait for streaming to complete
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Find code block
		const codeBlock = assistantMessage.locator('pre code');
		await expect(codeBlock).toBeVisible();

		// Verify syntax highlighting classes applied (highlight.js adds hljs class)
		const hasHighlightingClasses = await codeBlock.evaluate((el) => {
			return el.classList.contains('hljs') || el.className.includes('language-');
		});
		expect(hasHighlightingClasses).toBe(true);

		// Verify syntax highlighting COLORS are applied (spans with hljs- classes have color)
		const hasColoredSyntax = await codeBlock.evaluate((el) => {
			const spans = el.querySelectorAll('span[class*="hljs-"]');
			if (spans.length === 0) return false;

			// Check that at least one span has a non-default color
			for (const span of spans) {
				const color = window.getComputedStyle(span).color;
				// github-dark theme uses colors like rgb(255, 123, 114) for keywords
				// Check it's not just white/gray default text
				if (color && !color.includes('255, 255, 255') && !color.includes('228, 228, 231')) {
					return true;
				}
			}
			return false;
		});
		expect(hasColoredSyntax).toBe(true);

		// Verify code block stays within message bubble (no overflow)
		const preElement = assistantMessage.locator('pre');
		const messageBounds = await assistantMessage.boundingBox();
		const preBounds = await preElement.boundingBox();

		if (messageBounds && preBounds) {
			// Code block should not extend beyond message bubble (with small tolerance for borders)
			expect(preBounds.x).toBeGreaterThanOrEqual(messageBounds.x - 1);
			expect(preBounds.x + preBounds.width).toBeLessThanOrEqual(
				messageBounds.x + messageBounds.width + 1
			);
		}

		await page.screenshot({
			path: 'test-results/screenshots/code-block-highlighting.png',
			fullPage: true
		});
	});

	test('preserves context across multiple turns', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// First message: establish context
		await textarea.fill('My name is TestUser. Remember that name.');
		await sendButton.click();

		// Wait for first response
		const firstResponse = page.locator('.bg-muted').first();
		await expect(firstResponse).toBeVisible({ timeout: 30000 });

		// Wait for input to be re-enabled (streaming complete)
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Second message: query context
		await textarea.fill('What is my name?');
		await sendButton.click();

		// Wait for second response to appear
		const secondResponse = page.locator('.bg-muted').nth(1);
		await expect(secondResponse).toBeVisible({ timeout: 30000 });

		// Wait for streaming to complete (input re-enabled)
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Verify it remembers the name
		const responseText = await secondResponse.textContent();
		expect(responseText?.toLowerCase()).toContain('testuser');

		await page.screenshot({
			path: 'test-results/screenshots/chat-context.png',
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

		// Input should be disabled during streaming
		await expect(textarea).toBeDisabled({ timeout: 5000 });

		// Wait for streaming to complete
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });

		// Input should be re-enabled after response
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
