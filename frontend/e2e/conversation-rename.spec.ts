import { test, expect } from './fixtures';

// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
test.describe.skip('Conversation Rename', () => {
	test.setTimeout(60000);

	test('edit button appears on hover over conversation item', async ({ page }) => {
		await page.goto('/');

		// Create a conversation first
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find conversation item in sidebar
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();

		// Hover over the item to show edit button (triggers CSS transition)
		await conversationItem.hover();

		// Edit button should become visible after hover (wait for CSS transition) - scoped to item
		const editButton = conversationItem.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });

		await page.screenshot({
			path: 'test-results/screenshots/rename-edit-button-hover.png',
			fullPage: true
		});
	});

	test('clicking edit button enters rename mode', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "hello" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find and hover over the conversation item
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		await conversationItem.hover();

		// Wait for edit button to be visible (CSS transition) - scoped to item
		const editButton = conversationItem.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });

		// Click edit button
		await editButton.click();

		// Rename input should appear
		const renameInput = page.getByTestId('rename-input');
		await expect(renameInput).toBeVisible();
		await expect(renameInput).toBeFocused();

		await page.screenshot({
			path: 'test-results/screenshots/rename-editing.png',
			fullPage: true
		});
	});

	test('double-click on conversation title enters rename mode', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "world" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find the conversation title span and double-click
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		const titleSpan = conversationItem.locator('span.truncate');

		await titleSpan.dblclick();

		// Rename input should appear
		const renameInput = page.getByTestId('rename-input');
		await expect(renameInput).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/rename-double-click.png',
			fullPage: true
		});
	});

	test('Enter key saves new title', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "rename test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Enter rename mode
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		await conversationItem.hover();
		const editButton = conversationItem.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Type new title
		const renameInput = page.getByTestId('rename-input');
		await renameInput.clear();
		await renameInput.fill('My Custom Title');

		// Press Enter to save
		await renameInput.press('Enter');

		// Input should disappear and new title should be visible
		await expect(renameInput).not.toBeVisible();
		await expect(page.locator('aside')).toContainText('My Custom Title');

		await page.screenshot({
			path: 'test-results/screenshots/rename-saved.png',
			fullPage: true
		});
	});

	test('Escape key cancels rename', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "escape test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Get original title
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		const originalTitle = await conversationItem.locator('span.truncate').textContent();

		// Enter rename mode
		await conversationItem.hover();
		const editButton = conversationItem.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Type different title
		const renameInput = page.getByTestId('rename-input');
		await renameInput.clear();
		await renameInput.fill('Should Not Save');

		// Press Escape to cancel
		await renameInput.press('Escape');

		// Input should disappear and original title should remain
		await expect(renameInput).not.toBeVisible();
		await expect(conversationItem.locator('span.truncate')).toHaveText(originalTitle || '');

		await page.screenshot({
			path: 'test-results/screenshots/rename-cancelled.png',
			fullPage: true
		});
	});

	test('blur saves new title', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "blur test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Enter rename mode
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();
		await conversationItem.hover();
		const editButton = conversationItem.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Type new title
		const renameInput = page.getByTestId('rename-input');
		await renameInput.clear();
		await renameInput.fill('Blur Saved Title');

		// Click elsewhere to blur
		await textarea.click();

		// New title should be saved
		await expect(renameInput).not.toBeVisible();
		await expect(page.locator('aside')).toContainText('Blur Saved Title');

		await page.screenshot({
			path: 'test-results/screenshots/rename-blur-saved.png',
			fullPage: true
		});
	});
});
