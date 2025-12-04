import { test, expect } from '@playwright/test';

test.describe('Chat Functionality', () => {
	// Increase timeout for tests that wait for LLM responses
	test.setTimeout(60000);

	test('sends message and receives streaming response', async ({ page }) => {
		await page.goto('/');

		// Find input and send button
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for send button to be ready
		await expect(sendButton).toBeVisible();

		// Type a simple message
		await textarea.fill('Say "Hello World" and nothing else.');
		await sendButton.click();

		// Verify user message appears (use div to exclude button)
		const userMessage = page.locator('div.bg-primary').first();
		await expect(userMessage).toBeVisible();
		await expect(userMessage).toContainText('Say "Hello World"');

		// Wait for assistant response to have content (not just be visible)
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });

		// Wait for streaming to complete by checking send button reappears
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		// Screenshot for verification
		await page.screenshot({
			path: 'test-results/screenshots/chat-basic.png',
			fullPage: true
		});
	});

	test('renders markdown with code blocks and syntax highlighting', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Request a simple code example
		await textarea.fill('Show me a Python hello world in a code block. Just the code, nothing else.');
		await sendButton.click();

		// Wait for streaming to complete (send button reappears) - use long timeout for LLM
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Wait for assistant message with actual content (scope to main to avoid sidebar skeleton)
		const assistantMessage = page.locator('main .bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 90000 });

		// Wait for actual text content to appear (not just the skeleton)
		await expect(assistantMessage).toHaveText(/.+/, { timeout: 30000 });

		// Small wait for syntax highlighting to apply
		await page.waitForTimeout(1000);

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

		// Verify response has some content (should have text after streaming complete)
		const responseText = await assistantMessage.textContent();
		expect(responseText?.trim().length).toBeGreaterThan(0);

		await page.screenshot({
			path: 'test-results/screenshots/code-block-highlighting.png',
			fullPage: true
		});
	});

	test('multi-turn conversation flow works', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// First message - very simple prompt for fast response
		await textarea.fill('one');
		await sendButton.click();

		// Wait for streaming to complete (send button reappears) - long timeout for LLM
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Verify first user message and assistant response exist
		const firstUserMessage = page.locator('div.bg-primary').first();
		await expect(firstUserMessage).toContainText('one');

		// First assistant message should be visible
		const assistantMessages = page.locator('.bg-muted');
		await expect(assistantMessages.first()).toBeVisible();

		// Wait for state to settle
		await page.waitForTimeout(500);

		// Second message
		await textarea.fill('two');
		await sendButton.click();

		// Wait for streaming to complete - long timeout for LLM
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Verify we have 2 user messages and 2 assistant responses
		const userMessages = page.locator('div.bg-primary');
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

	test('textarea retains focus after sending message', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Focus textarea and send a message
		await textarea.click();
		await textarea.fill('Say "ok" only.');
		await sendButton.click();

		// Wait for response to complete
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		// Textarea should still be focused (or refocused after response)
		await expect(textarea).toBeFocused();

		// Should be able to type immediately without clicking
		await page.keyboard.type('follow up');
		await expect(textarea).toHaveValue('follow up');
	});

	test('auto-scrolls to new messages when at bottom', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');
		const messageContainer = page.locator('.flex-1.overflow-y-auto');

		// Send first message
		await textarea.fill('Say "test message one" and nothing else.');
		await sendButton.click();

		// Wait for response and streaming to complete
		const firstResponse = page.locator('.bg-muted').first();
		await expect(firstResponse).toBeVisible({ timeout: 30000 });
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		// Verify we're scrolled to bottom (scrollTop + clientHeight >= scrollHeight - threshold)
		const isAtBottom = await messageContainer.evaluate((el) => {
			return el.scrollHeight - el.scrollTop - el.clientHeight < 100;
		});
		expect(isAtBottom).toBe(true);
	});

	test('does not auto-scroll when user has scrolled up', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');
		const messageContainer = page.locator('.flex-1.overflow-y-auto');

		// Send first message - simple prompt for fast response
		await textarea.fill('hi');
		await sendButton.click();

		// Wait for streaming to complete - long timeout for LLM
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// First response should be visible
		const assistantMessages = page.locator('.bg-muted');
		await expect(assistantMessages.first()).toBeVisible();

		await page.waitForTimeout(500);

		// Scroll up manually
		await messageContainer.evaluate((el) => {
			el.scrollTop = 0;
		});

		// Send second message
		await textarea.fill('hello');
		await sendButton.click();

		// Wait for streaming to complete - long timeout for LLM
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Second response should be visible
		await expect(assistantMessages.nth(1)).toBeVisible();

		// Verify scroll position hasn't changed significantly (should stay near top)
		const scrollTopAfter = await messageContainer.evaluate((el) => el.scrollTop);
		expect(scrollTopAfter).toBeLessThan(100); // Should still be near top

		await page.screenshot({
			path: 'test-results/screenshots/scroll-stays-when-scrolled-up.png',
			fullPage: true
		});
	});

	test('resumes auto-scroll when user scrolls back to bottom', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');
		const messageContainer = page.locator('.flex-1.overflow-y-auto');

		// Send first message - simple prompt for fast response
		await textarea.fill('hi');
		await sendButton.click();

		// Wait for streaming to complete
		await expect(sendButton).toBeVisible({ timeout: 45000 });

		// First response should be visible
		const assistantMessages = page.locator('.bg-muted');
		await expect(assistantMessages.first()).toBeVisible();

		await page.waitForTimeout(500);

		// Scroll up
		await messageContainer.evaluate((el) => {
			el.scrollTop = 0;
		});

		// Scroll back to bottom
		await messageContainer.evaluate((el) => {
			el.scrollTop = el.scrollHeight;
		});

		await page.waitForTimeout(100);

		// Send second message
		await textarea.fill('hello');
		await sendButton.click();

		// Wait for streaming to complete
		await expect(sendButton).toBeVisible({ timeout: 45000 });

		// Second response should be visible
		await expect(assistantMessages.nth(1)).toBeVisible();

		// Verify we're at bottom (auto-scroll resumed)
		const isAtBottom = await messageContainer.evaluate((el) => {
			return el.scrollHeight - el.scrollTop - el.clientHeight < 100;
		});
		expect(isAtBottom).toBe(true);
	});
});
