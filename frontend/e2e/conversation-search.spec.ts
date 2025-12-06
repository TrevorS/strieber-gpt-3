import { test, expect } from '@playwright/test';

test.describe('Conversation Search', () => {
	test.setTimeout(90000);

	test('search input filters conversations', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Create first conversation
		await textarea.fill('Say "apple" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Create second conversation with different topic
		await page.locator('aside button:has-text("New Chat")').click();
		await expect(page).toHaveURL('/');

		await textarea.fill('Say "banana" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Wait for conversation to appear in sidebar
		await expect(page.locator('aside')).toContainText('banana', { timeout: 10000 });

		// Find search input and filter
		const searchInput = page.locator('aside input[placeholder="Search conversations..."]');
		await expect(searchInput).toBeVisible();

		// Search for "apple"
		await searchInput.fill('apple');

		// Should show only the apple conversation
		await expect(page.locator('aside')).toContainText('apple');
		// Banana should be filtered out
		await expect(page.locator('aside')).not.toContainText('banana');

		await page.screenshot({
			path: 'test-results/screenshots/search-filtered.png',
			fullPage: true
		});
	});

	test('empty results shows "No matching conversations"', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Create a conversation first
		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Search for something that doesn't exist
		const searchInput = page.locator('aside input[placeholder="Search conversations..."]');
		await searchInput.fill('xyznonexistent123');

		// Should show no matching conversations message
		await expect(page.locator('aside')).toContainText('No matching conversations');

		await page.screenshot({
			path: 'test-results/screenshots/search-no-results.png',
			fullPage: true
		});
	});

	test('clearing search restores full list', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Create first conversation
		await textarea.fill('Say "first" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Create second conversation
		await page.locator('aside button:has-text("New Chat")').click();
		await expect(page).toHaveURL('/');

		await textarea.fill('Say "second" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// Wait for second conversation to appear
		await expect(page.locator('aside')).toContainText('second', { timeout: 10000 });

		const searchInput = page.locator('aside input[placeholder="Search conversations..."]');

		// Filter to show only first
		await searchInput.fill('first');
		await expect(page.locator('aside')).not.toContainText('second');

		// Clear the search
		await searchInput.clear();

		// Both should be visible again
		await expect(page.locator('aside')).toContainText('first');
		await expect(page.locator('aside')).toContainText('second');

		await page.screenshot({
			path: 'test-results/screenshots/search-cleared.png',
			fullPage: true
		});
	});
});
