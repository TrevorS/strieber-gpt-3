import { test, expect } from './fixtures';

// SKIP: SSE streaming unreliable in Docker E2E environment - message edit requires completed streaming
test.describe.skip('Message Edit', () => {
	test.setTimeout(90000);

	test('edit button appears on hover for user messages', async ({ page }) => {
		await page.goto('/');

		// Send a message first
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find user message container and hover
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await userMessage.hover();

		// Edit button should become visible on hover (wait for CSS transition)
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await expect(editButton).toContainText('Edit');

		await page.screenshot({
			path: 'test-results/screenshots/edit-button-hover.png',
			fullPage: true
		});
	});

	test('clicking edit button enters edit mode with textarea', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Original message text.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Hover and click edit button
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await userMessage.hover();
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Edit textarea should appear with original content
		const editTextarea = page.getByTestId('edit-textarea');
		await expect(editTextarea).toBeVisible();
		await expect(editTextarea).toBeFocused();
		await expect(editTextarea).toHaveValue('Original message text.');

		// Save and cancel buttons should be visible
		await expect(page.getByTestId('save-button')).toBeVisible();
		await expect(page.getByTestId('cancel-button')).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/edit-mode.png',
			fullPage: true
		});
	});

	test('save button triggers message update', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Original content here.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Enter edit mode
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await userMessage.hover();
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Modify content
		const editTextarea = page.getByTestId('edit-textarea');
		await editTextarea.clear();
		await editTextarea.fill('Updated content here.');

		// Click save
		await page.getByTestId('save-button').click();

		// Edit mode should close
		await expect(editTextarea).not.toBeVisible();

		// New content should be visible
		await expect(userMessage).toContainText('Updated content here.');

		await page.screenshot({
			path: 'test-results/screenshots/edit-saved.png',
			fullPage: true
		});
	});

	test('cancel button restores original content', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Keep this original.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Enter edit mode
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await userMessage.hover();
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Modify content
		const editTextarea = page.getByTestId('edit-textarea');
		await editTextarea.clear();
		await editTextarea.fill('This should not save.');

		// Click cancel
		await page.getByTestId('cancel-button').click();

		// Edit mode should close
		await expect(editTextarea).not.toBeVisible();

		// Original content should be restored
		await expect(userMessage).toContainText('Keep this original.');
		await expect(userMessage).not.toContainText('This should not save.');

		await page.screenshot({
			path: 'test-results/screenshots/edit-cancelled.png',
			fullPage: true
		});
	});

	test('Escape key cancels edit', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Escape test message.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Enter edit mode
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await userMessage.hover();
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		// Modify content
		const editTextarea = page.getByTestId('edit-textarea');
		await editTextarea.clear();
		await editTextarea.fill('Should not save on Escape.');

		// Press Escape
		await editTextarea.press('Escape');

		// Edit mode should close
		await expect(editTextarea).not.toBeVisible();

		// Original content should be restored
		await expect(userMessage).toContainText('Escape test message.');

		await page.screenshot({
			path: 'test-results/screenshots/edit-escape-cancelled.png',
			fullPage: true
		});
	});

	test('edited indicator shows after edit', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Before editing.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Initially, no "(edited)" indicator
		const userMessage = page.locator('.group:has(.bg-primary)').first();
		await expect(userMessage).not.toContainText('(edited)');

		// Enter edit mode and save different content
		await userMessage.hover();
		const editButton = userMessage.getByTestId('edit-button');
		await expect(editButton).toBeVisible({ timeout: 5000 });
		await editButton.click();

		const editTextarea = page.getByTestId('edit-textarea');
		await editTextarea.clear();
		await editTextarea.fill('After editing.');
		await page.getByTestId('save-button').click();

		// Should now show "(edited)" indicator
		await expect(userMessage).toContainText('(edited)');

		await page.screenshot({
			path: 'test-results/screenshots/edited-indicator.png',
			fullPage: true
		});
	});
});
