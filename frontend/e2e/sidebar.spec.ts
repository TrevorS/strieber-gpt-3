import { test, expect } from './fixtures';
import { setupLogCapture, filterLogs, expectLog } from './helpers/logger';

// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
test.describe.skip('Sidebar Navigation', () => {
	test.setTimeout(60000);

	test('New Chat button clears conversation in single click', async ({ page }) => {
		// Set up log capture before navigation
		const logs = setupLogCapture(page);

		await page.goto('/');

		// Send a message to create a conversation
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		await textarea.fill('Say "test" only.');
		await sendButton.click();

		// Wait for response and URL change
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 60000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Screenshot before clicking New Chat
		await page.screenshot({
			path: 'test-results/screenshots/new-chat-before.png',
			fullPage: true
		});

		// Click New Chat button (the one with Plus icon in sidebar header)
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();
		await newChatButton.click();

		// Verify URL is home
		await expect(page).toHaveURL('/');

		// Small wait to let any effects settle
		await page.waitForTimeout(100);

		// Verify no messages shown
		const userMessages = page.locator('div.bg-primary');
		const assistantMessages = page.locator('.bg-muted');
		await expect(userMessages).toHaveCount(0);
		await expect(assistantMessages).toHaveCount(0);

		// Verify input is ready
		await expect(textarea).toBeEnabled();
		await expect(textarea).toBeEmpty();

		// Verify structured logging captured key events
		// (Either "loaded" or "No saved" depending on test state)
		const persistenceLogs = filterLogs(logs, { category: 'persistence' });
		expect(persistenceLogs.length).toBeGreaterThan(0);

		expectLog(logs, { category: 'store', message: /Action: create/i });
		// handleNew triggers setActive(null) - verify this happens
		expectLog(logs, { category: 'store', message: /Action: setActive/i });

		// Verify a setActive log with null exists (new chat state)
		const setActiveNullLogs = filterLogs(logs, { category: 'store', message: /Action: setActive/ }).filter(
			(log) => log.data?.newActiveId === null
		);
		expect(setActiveNullLogs.length).toBeGreaterThan(0);

		await page.screenshot({
			path: 'test-results/screenshots/new-chat-single-click.png',
			fullPage: true
		});
	});

	test('New Chat allows immediate new message', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create first conversation
		await textarea.fill('Say "first" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		const firstUrl = page.url();

		// Click New Chat button (the one with Plus icon in sidebar header)
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();
		await newChatButton.click();
		await expect(page).toHaveURL('/');

		// Immediately send new message (should work without double-click)
		await textarea.fill('Say "second" only.');
		await sendButton.click();

		// Should navigate to NEW conversation URL
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		const secondUrl = page.url();
		expect(secondUrl).not.toBe(firstUrl);

		// Should only see the new message, not old conversation
		const userMessages = page.locator('div.bg-primary');
		await expect(userMessages).toHaveCount(1);
		await expect(userMessages.first()).toContainText('second');

		await page.screenshot({
			path: 'test-results/screenshots/new-chat-immediate-send.png',
			fullPage: true
		});
	});

	test('conversation switching and New Chat works correctly', async ({ page }) => {
		test.slow(); // Double timeout for multi-conversation test (2+ LLM roundtrips)
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();

		// Create first conversation
		await textarea.fill('First conversation message.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });
		const firstConversationUrl = page.url();

		// Click New Chat button
		await newChatButton.click();
		await expect(page).toHaveURL('/');

		// Create second conversation
		await textarea.fill('Second conversation message.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Click on first conversation in sidebar (second item, after the current one)
		// Conversations are sorted by most recent first, so first conv is second in list
		const conversationItems = page.locator('aside').locator('button:has-text("New Chat")');
		// Skip the header button, then skip current conversation, click the older one
		await conversationItems.nth(2).click();

		// Should navigate back to first conversation
		await expect(page).toHaveURL(firstConversationUrl);

		// Verify first conversation messages shown
		const userMessages = page.locator('div.bg-primary');
		await expect(userMessages.first()).toContainText('First conversation');

		// Click New Chat again
		await newChatButton.click();
		await expect(page).toHaveURL('/');

		// Verify empty state
		await expect(userMessages).toHaveCount(0);

		await page.screenshot({
			path: 'test-results/screenshots/conversation-switching.png',
			fullPage: true
		});
	});
});
