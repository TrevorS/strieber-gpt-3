import { test, expect } from '@playwright/test';

test.describe('Delete Conversation', () => {
	test.setTimeout(60000);

	test('delete button appears on hover over conversation item', async ({ page }) => {
		await page.goto('/');

		// Create a conversation first
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Find conversation item in sidebar (has delete button inside)
		const conversationItem = page.locator('aside button:has([data-testid="delete-button"])').first();

		// Hover over the item to show delete button
		await conversationItem.hover();

		// Delete button should be visible on hover
		const deleteButton = conversationItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/delete-button-hover.png',
			fullPage: true
		});
	});

	test('deleting active conversation navigates to home', async ({ page }) => {
		await page.goto('/');

		// Create a conversation
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Find and hover over the conversation item
		const conversationItem = page.locator('aside button:has([data-testid="delete-button"])').first();
		await conversationItem.hover();

		// Click delete button
		await conversationItem.getByTestId('delete-button').click();

		// Should navigate to home
		await expect(page).toHaveURL('/');

		// Conversation list should show "No conversations yet"
		await expect(page.locator('aside')).toContainText('No conversations yet');

		await page.screenshot({
			path: 'test-results/screenshots/delete-active-conversation.png',
			fullPage: true
		});
	});

	test('deleting non-active conversation stays on current conversation', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.locator('button[type="submit"], button:has(svg)').last();

		// Create first conversation
		await textarea.fill('First conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });
		const firstUrl = page.url();

		// Create second conversation via New Chat
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		await textarea.fill('Second conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });
		const secondUrl = page.url();

		// We're on second conversation, delete the first (non-active) one
		// The first conversation should be the second item in the list (older)
		const conversationItems = page.locator('aside button:has([data-testid="delete-button"])');

		// The second item is the older conversation
		const olderConversation = conversationItems.nth(1);
		await olderConversation.hover();
		await olderConversation.getByTestId('delete-button').click();

		// Should still be on second conversation
		await expect(page).toHaveURL(secondUrl);

		// Should only have one conversation left
		await expect(conversationItems).toHaveCount(1);

		await page.screenshot({
			path: 'test-results/screenshots/delete-non-active-conversation.png',
			fullPage: true
		});
	});

	test('deleting last active conversation navigates to home', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Create two conversations
		await textarea.fill('First conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');
		await textarea.fill('Second conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(sendButton).toBeVisible({ timeout: 30000 });

		// Delete both conversations
		let conversationItems = page.locator('aside button:has([data-testid="delete-button"])');

		// Delete first (current)
		const firstItem = conversationItems.first();
		await firstItem.hover();
		await firstItem.getByTestId('delete-button').click();

		// Should navigate to home or another conversation
		await page.waitForTimeout(500);

		// Delete any remaining conversations
		conversationItems = page.locator('aside button:has([data-testid="delete-button"])');
		const remainingCount = await conversationItems.count();

		if (remainingCount > 0) {
			const lastItem = conversationItems.first();
			await lastItem.hover();
			await lastItem.getByTestId('delete-button').click();
		}

		// Should navigate to home with empty state
		await expect(page).toHaveURL('/');
		await expect(page.locator('aside')).toContainText('No conversations yet');

		await page.screenshot({
			path: 'test-results/screenshots/delete-all-conversations.png',
			fullPage: true
		});
	});

	test('delete button works on mobile after opening sidebar', async ({ page }) => {
		await page.setViewportSize({ width: 375, height: 667 });
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		// Create a conversation
		await textarea.fill('Test mobile delete.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Open sidebar
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();

		// Find conversation item and hover/tap
		const conversationItem = sidebar.locator('button:has([data-testid="delete-button"])').first();
		await conversationItem.hover();

		// Delete button should be visible
		const deleteButton = conversationItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible();

		// Click delete
		await deleteButton.click();

		// Wait for navigation
		await page.waitForTimeout(350);

		// Should navigate to home
		await expect(page).toHaveURL('/');

		await page.screenshot({
			path: 'test-results/screenshots/delete-mobile.png',
			fullPage: true
		});
	});
});
