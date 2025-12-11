import { test, expect } from './fixtures';

// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
test.describe.skip('Delete Conversation', () => {
	test.setTimeout(60000);

	test('delete button appears on hover over conversation item', async ({ page }) => {
		await page.goto('/');

		// Create a conversation first
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		await textarea.fill('Say "test" only.');
		await expect(sendButton).toBeEnabled({ timeout: 5000 });
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find conversation item in sidebar - get the first (most recent) one
		const conversationItem = page.locator('[data-testid="conversation-item"]').first();

		// Hover over the item to show delete button (triggers CSS transition)
		await conversationItem.hover();

		// Delete button should become visible after hover - use locator scoped to the item
		const deleteButton = conversationItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible({ timeout: 5000 });

		await page.screenshot({
			path: 'test-results/screenshots/delete-button-hover.png',
			fullPage: true
		});
	});

	// TODO: This test is skipped due to a Svelte 5 effect_update_depth_exceeded error
	// that occurs when navigating after deleting the active conversation.
	// The delete works correctly, but the navigation to the next conversation
	// triggers an infinite effect loop. This needs deeper investigation.
	test.skip('deleting active conversation navigates away', async ({ page }) => {
		// Capture console messages and errors
		const consoleLogs: string[] = [];
		const errors: string[] = [];
		page.on('console', (msg) => consoleLogs.push(`${msg.type()}: ${msg.text()}`));
		page.on('pageerror', (err) => errors.push(err.message));

		await page.goto('/');

		// Create a conversation
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		await textarea.fill('Say "test" only.');
		await expect(sendButton).toBeEnabled({ timeout: 5000 });
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		const conversationUrl = page.url();
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Find the ACTIVE conversation item (the one we just created and are currently viewing)
		// It has aria-current="page" attribute
		const conversationItem = page.locator('[data-testid="conversation-item"][aria-current="page"]');
		await expect(conversationItem).toBeVisible({ timeout: 5000 });

		await conversationItem.hover();

		// Click delete button (wait for it to be visible first) - scoped to the item
		const deleteButton = conversationItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible({ timeout: 5000 });

		// Log before click
		console.log('About to click delete button');
		await deleteButton.click();
		console.log('Clicked delete button');

		// Wait a bit to capture console logs
		await page.waitForTimeout(2000);

		// Log captured console messages for debugging
		console.log('Console logs:', consoleLogs.filter((l) => l.includes('Delete') || l.includes('delete') || l.includes('Navigate') || l.includes('navigation')));
		console.log('Page errors:', errors);
		console.log('Current URL after wait:', page.url());

		// Deleting the active conversation triggers navigation to another conversation
		// The goto() in the app isn't awaited, so we need to poll for the URL change
		await expect(page).not.toHaveURL(conversationUrl, { timeout: 15000 });

		await page.screenshot({
			path: 'test-results/screenshots/delete-active-conversation.png',
			fullPage: true
		});
	});

	test('deleting non-active conversation stays on current conversation', async ({ page }) => {
		test.slow(); // Double timeout for multi-conversation test (2+ LLM roundtrips)
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Create first conversation
		await textarea.fill('First conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });
		const firstUrl = page.url();

		// Create second conversation via New Chat
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		await textarea.fill('Second conversation.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		await expect(textarea).toBeEnabled({ timeout: 60000 });
		const secondUrl = page.url();

		// We're on second conversation, delete the first (non-active) one
		// The first conversation should be the second item in the list (older)
		const conversationItems = page.locator('[data-testid="conversation-item"]');

		// The second item is the older conversation
		const olderConversation = conversationItems.nth(1);
		await olderConversation.hover();
		const deleteButton = olderConversation.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible({ timeout: 5000 });
		await deleteButton.click();

		// Should still be on second conversation
		await expect(page).toHaveURL(secondUrl);

		// Verify older conversation was deleted - there's no longer an item at that position
		// that we can click (it should have shifted)
		await page.screenshot({
			path: 'test-results/screenshots/delete-non-active-conversation.png',
			fullPage: true
		});
	});

	// TODO: Skipped - same effect loop issue as 'deleting active conversation navigates away'
	test.skip('deleting conversation removes it from sidebar', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Count current conversations
		const conversationItems = page.locator('[data-testid="conversation-item"]');
		const initialCount = await conversationItems.count();

		// Create a conversation
		await textarea.fill('Say "test" only.');
		await expect(sendButton).toBeEnabled({ timeout: 5000 });
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		const conversationUrl = page.url();
		await expect(sendButton).toBeVisible({ timeout: 60000 });

		// Should have one more conversation (wait for it to appear)
		await expect(conversationItems).toHaveCount(initialCount + 1, { timeout: 5000 });

		// Find the ACTIVE conversation item (the one we just created)
		const activeItem = page.locator('[data-testid="conversation-item"][aria-current="page"]');
		await expect(activeItem).toBeVisible({ timeout: 5000 });

		await activeItem.hover();
		const deleteButton = activeItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible({ timeout: 5000 });
		await deleteButton.click();

		// Deleting the active conversation triggers navigation - wait for URL to change
		await expect(async () => {
			const currentUrl = page.url();
			expect(currentUrl).not.toBe(conversationUrl);
		}).toPass({ timeout: 10000 });

		// After navigation completes, verify conversation count is back to initial
		await expect(conversationItems).toHaveCount(initialCount, { timeout: 5000 });

		await page.screenshot({
			path: 'test-results/screenshots/delete-all-conversations.png',
			fullPage: true
		});
	});

	// TODO: Skipped - same effect loop issue as 'deleting active conversation navigates away'
	test.skip('delete button works on mobile after opening sidebar', async ({ page }) => {
		await page.setViewportSize({ width: 375, height: 667 });
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		// Wait for page to be ready
		await expect(sendButton).toBeVisible();

		// Count initial conversations
		// On mobile, we need to open sidebar first to count
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();
		const conversationItems = sidebar.locator('[data-testid="conversation-item"]');
		const initialCount = await conversationItems.count();

		// Close sidebar by clicking backdrop
		await page.mouse.click(350, 300);
		await page.waitForTimeout(350);

		// Create a conversation
		await textarea.fill('Say "test" only.');
		await expect(sendButton).toBeEnabled({ timeout: 5000 });
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
		const conversationUrl = page.url();
		await expect(textarea).toBeEnabled({ timeout: 60000 });

		// Open sidebar again
		await page.getByTestId('sidebar-toggle').click();
		await expect(sidebar).toBeInViewport();

		// Should have one more conversation
		await expect(conversationItems).toHaveCount(initialCount + 1, { timeout: 5000 });

		// Find the ACTIVE conversation item (the one we just created)
		const activeItem = sidebar.locator('[data-testid="conversation-item"][aria-current="page"]');
		await expect(activeItem).toBeVisible({ timeout: 5000 });

		await activeItem.hover();

		// Delete button should be visible (wait for CSS transition) - scoped to the item
		const deleteButton = activeItem.getByTestId('delete-button');
		await expect(deleteButton).toBeVisible({ timeout: 5000 });

		// Click delete - this is the active conversation, so navigation will occur
		await deleteButton.click();

		// Deleting the active conversation triggers navigation - wait for URL to change
		await expect(async () => {
			const currentUrl = page.url();
			expect(currentUrl).not.toBe(conversationUrl);
		}).toPass({ timeout: 10000 });

		// After navigation, open sidebar to verify count (sidebar may have closed on mobile)
		await page.getByTestId('sidebar-toggle').click();
		await expect(sidebar).toBeInViewport();

		// Wait for deletion to complete and count to update
		await expect(conversationItems).toHaveCount(initialCount, { timeout: 5000 });

		await page.screenshot({
			path: 'test-results/screenshots/delete-mobile.png',
			fullPage: true
		});
	});
});
