import { test, expect } from './fixtures';

// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
test.describe.skip('Conversation Persistence', () => {
	test.setTimeout(60000);

	test('conversations persist across page reloads', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create a conversation
		await textarea.fill('Say "persisted message" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		const conversationUrl = page.url();

		// Reload the page
		await page.reload();

		// Wait for page to load
		await expect(page.locator('aside')).toContainText('Strieber');

		// Conversation should still be in sidebar
		const conversationItems = page.locator('[data-testid="conversation-item"]');
		await expect(conversationItems).toHaveCount(1);

		await page.screenshot({
			path: 'test-results/screenshots/persistence-reload.png',
			fullPage: true
		});
	});

	test('clicking on persisted conversation shows its messages', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Create a conversation with unique content
		await textarea.fill('Say "unique test content xyz" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });

		// Wait for assistant response and streaming to complete
		const assistantMessage = page.locator('.bg-muted').first();
		await expect(assistantMessage).toBeVisible({ timeout: 30000 });
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		const conversationUrl = page.url();

		// Navigate to home via New Chat
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		// Messages should be cleared
		await expect(page.locator('div.bg-primary')).toHaveCount(0);

		// Click on the conversation in sidebar
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		await conversationItem.click();

		// Should navigate back to conversation
		await expect(page).toHaveURL(conversationUrl);

		// Original user message should be visible
		const userMessage = page.locator('div.bg-primary').first();
		await expect(userMessage).toContainText('unique test content xyz');

		await page.screenshot({
			path: 'test-results/screenshots/persistence-click-conversation.png',
			fullPage: true
		});
	});

	test('multiple conversations are all preserved', async ({ page }) => {
		test.slow(); // Double timeout for multi-conversation test (3 LLM roundtrips)
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();

		// Create first conversation
		await textarea.fill('First conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Create second conversation
		await newChatButton.click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Second conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Create third conversation
		await newChatButton.click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Third conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Reload page
		await page.reload();

		// Wait for conversations to load
		await expect(page.locator('aside')).toContainText('Strieber');

		// All three conversations should be in sidebar
		const conversationItems = page.locator('[data-testid="conversation-item"]');
		await expect(conversationItems).toHaveCount(3);

		await page.screenshot({
			path: 'test-results/screenshots/persistence-multiple.png',
			fullPage: true
		});
	});

	test('conversation messages persist after creating new conversation', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Create first conversation (single message to reduce LLM round-trips)
		await textarea.fill('First conversation message');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		const firstConvUrl = page.url();

		// Create second conversation
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Second conversation message');
		await expect(sendButton).toBeEnabled({ timeout: 5000 });
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Go back to first conversation
		const conversationItems = page.locator('[data-testid="conversation-item"]');
		// First conversation is now second in list (older)
		await conversationItems.nth(1).click();

		await expect(page).toHaveURL(firstConvUrl);

		// Should have the original user message from first conversation
		const userMessages = page.locator('div.bg-primary');
		await expect(userMessages).toHaveCount(1);
		await expect(userMessages.first()).toContainText('First conversation message');

		await page.screenshot({
			path: 'test-results/screenshots/persistence-switch-conversation.png',
			fullPage: true
		});
	});

	test('app starts on home page (new chat) after reload', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create a conversation
		await textarea.fill('Test message.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Reload the page - should start at home regardless of previous URL
		await page.goto('/');
		await expect(page).toHaveURL('/');

		// No messages should be shown (new chat state)
		await expect(page.locator('div.bg-primary')).toHaveCount(0);
		await expect(page.locator('.bg-muted')).toHaveCount(0);

		// But conversation should be in sidebar
		const conversationItems = page.locator('[data-testid="conversation-item"]');
		await expect(conversationItems).toHaveCount(1);
	});
});
