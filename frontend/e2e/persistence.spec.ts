import { test, expect } from '@playwright/test';

test.describe('Conversation Persistence', () => {
	test.setTimeout(60000);

	test('conversations persist across page reloads', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create a conversation
		await textarea.fill('Say "persisted message" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		const conversationUrl = page.url();

		// Reload the page
		await page.reload();

		// Wait for page to load
		await expect(page.locator('aside')).toContainText('Strieber');

		// Conversation should still be in sidebar
		const conversationItems = page.locator('aside button:has([data-testid="delete-button"])');
		await expect(conversationItems).toHaveCount(1);

		await page.screenshot({
			path: 'test-results/screenshots/persistence-reload.png',
			fullPage: true
		});
	});

	test('clicking on persisted conversation shows its messages', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Create a conversation with unique content
		await textarea.fill('Say "unique test content xyz" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);

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
		const conversationItem = page.locator('aside button:has([data-testid="delete-button"])').first();
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
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();

		// Create first conversation
		await textarea.fill('First conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Create second conversation
		await newChatButton.click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Second conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Create third conversation
		await newChatButton.click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Third conversation content.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Reload page
		await page.reload();

		// Wait for conversations to load
		await expect(page.locator('aside')).toContainText('Strieber');

		// All three conversations should be in sidebar
		const conversationItems = page.locator('aside button:has([data-testid="delete-button"])');
		await expect(conversationItems).toHaveCount(3);

		await page.screenshot({
			path: 'test-results/screenshots/persistence-multiple.png',
			fullPage: true
		});
	});

	test('conversation messages persist after creating new conversation', async ({ page }) => {
		test.slow(); // Double timeout for LLM-dependent test
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Create first conversation with multi-turn (simple messages for speed)
		await textarea.fill('msg1');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		await page.waitForTimeout(500);

		await textarea.fill('msg2');
		await sendButton.click();
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		const firstConvUrl = page.url();

		// Create second conversation
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');
		await textarea.fill('msg3');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(sendButton).toBeVisible({ timeout: 90000 });

		// Go back to first conversation
		const conversationItems = page.locator('aside button:has([data-testid="delete-button"])');
		// First conversation is now second in list (older)
		await conversationItems.nth(1).click();

		await expect(page).toHaveURL(firstConvUrl);

		// Should have 2 user messages from original conversation
		const userMessages = page.locator('div.bg-primary');
		await expect(userMessages).toHaveCount(2);
		await expect(userMessages.first()).toContainText('msg1');
		await expect(userMessages.nth(1)).toContainText('msg2');

		await page.screenshot({
			path: 'test-results/screenshots/persistence-multi-turn.png',
			fullPage: true
		});
	});

	test('app starts on home page (new chat) after reload', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create a conversation
		await textarea.fill('Test message.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Reload the page - should start at home regardless of previous URL
		await page.goto('/');
		await expect(page).toHaveURL('/');

		// No messages should be shown (new chat state)
		await expect(page.locator('div.bg-primary')).toHaveCount(0);
		await expect(page.locator('.bg-muted')).toHaveCount(0);

		// But conversation should be in sidebar
		const conversationItems = page.locator('aside button:has([data-testid="delete-button"])');
		await expect(conversationItems).toHaveCount(1);
	});
});
