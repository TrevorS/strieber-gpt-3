/**
 * E2E Tests for Conversation API Flow
 *
 * Debug tests to verify conversation creation, persistence, and navigation
 * after the Conversations API integration.
 */
import { test, expect } from './fixtures';
import { setupLogCapture, filterLogs, waitForLog, printLogs } from './helpers/logger';

test.describe('Conversation API Flow', () => {
	// These tests interact with LLM, so give them more time
	test.setTimeout(60000);

	test('should load conversations from API on page load', async ({ page }) => {
		const logCapture = setupLogCapture(page);
		await page.goto('/');

		// Wait for fetchAll to complete
		await waitForLog(
			page,
			logCapture,
			{
				category: 'persistence',
				message: /Conversations loaded/
			},
			10000
		);

		// Screenshot the initial state
		await page.screenshot({ path: 'test-results/screenshots/api-flow-01-initial-load.png' });

		// Verify API was called (check logs)
		const apiLogs = filterLogs(logCapture, { category: 'api' });
		console.log('API logs on load:', apiLogs);
		printLogs(logCapture);

		// Should have made a GET request to /conversations
		expect(apiLogs.some((log) => log.message.includes('GET') && log.message.includes('conversations'))).toBe(true);
	});

	test('should create conversation and navigate on first message', async ({ page }) => {
		const logCapture = setupLogCapture(page);
		await page.goto('/');

		// Wait for initial load
		await page.waitForTimeout(1000);

		// Type and send a message
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		await textarea.fill('Hello, this is a test message');
		await page.screenshot({ path: 'test-results/screenshots/api-flow-02-message-typed.png' });

		await page.getByTestId('send-button').click();

		// URL should change to /c/{conv_id}
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });
		await page.screenshot({ path: 'test-results/screenshots/api-flow-03-navigated-to-conversation.png' });

		// Print logs to see what happened
		printLogs(logCapture);

		// Verify a conversation was created via API
		const apiLogs = filterLogs(logCapture, { category: 'api' });
		expect(apiLogs.some((log) => log.message.includes('POST') && log.message.includes('conversations'))).toBe(true);

		// Sidebar should show the new conversation
		const sidebar = page.locator('aside');
		await expect(sidebar).toBeVisible();
	});

	test('should persist conversation after page refresh', async ({ page }) => {
		const logCapture = setupLogCapture(page);

		// Create a conversation first
		await page.goto('/');
		await page.waitForTimeout(1000);

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		await textarea.fill('Test message for persistence');
		await page.getByTestId('send-button').click();

		// Wait for navigation
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });
		const currentUrl = page.url();

		// Wait for streaming to complete (send button visible again)
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 60000 });

		await page.screenshot({ path: 'test-results/screenshots/api-flow-04-before-refresh.png' });

		// Check messages before refresh
		const messagesBefore = await page.locator('main .bg-primary, main .bg-muted').count();
		console.log('Messages before refresh:', messagesBefore);

		// Refresh the page
		await page.reload();

		// Wait for conversations to load
		await waitForLog(
			page,
			logCapture,
			{
				category: 'persistence',
				message: /Conversations loaded/
			},
			10000
		);

		// Wait for items to be loaded for the conversation
		// The effect runs after isLoading becomes false and loads items
		await waitForLog(
			page,
			logCapture,
			{
				category: 'persistence',
				message: /Items loaded for conversation/
			},
			10000
		);

		await page.screenshot({ path: 'test-results/screenshots/api-flow-05-after-refresh.png' });

		// URL should still be the same
		await expect(page).toHaveURL(currentUrl);

		// Check messages after refresh - messages should now be loaded
		const messagesAfter = await page.locator('main .bg-primary, main .bg-muted').count();
		console.log('Messages after refresh:', messagesAfter);

		// Conversation should still be in sidebar
		const sidebar = page.locator('aside');
		const convButtons = sidebar.locator('button').filter({ hasNot: page.locator(':has-text("New Chat")') });
		await expect(convButtons.first()).toBeVisible();

		printLogs(logCapture);
	});

	// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
	test.skip('should create second conversation after refresh', async ({ page }) => {
		test.slow(); // Double timeout for multi-conversation test (2+ LLM roundtrips)
		const logCapture = setupLogCapture(page);

		// Start fresh, create first conversation
		await page.goto('/');
		await page.waitForTimeout(1000);

		await page.locator('textarea').fill('First conversation message');
		await page.getByTestId('send-button').click();
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });

		// Wait for response
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 30000 });

		await page.screenshot({ path: 'test-results/screenshots/api-flow-06-first-conv-created.png' });

		// Refresh
		await page.reload();
		await page.waitForTimeout(2000);

		await page.screenshot({ path: 'test-results/screenshots/api-flow-07-after-refresh.png' });

		// Click "New Chat" button (use role to get just the button, not the title spans)
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		await page.screenshot({ path: 'test-results/screenshots/api-flow-08-new-chat-clicked.png' });

		// Create second conversation
		await page.locator('textarea').fill('Second conversation message');
		await page.getByTestId('send-button').click();
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });

		// Wait for response
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 30000 });

		await page.screenshot({ path: 'test-results/screenshots/api-flow-09-second-conv-created.png' });

		// Count conversations in sidebar (should be 2)
		const sidebar = page.locator('aside');
		const convButtons = sidebar.locator('button').filter({ hasNot: page.locator(':has-text("New Chat")') });
		const count = await convButtons.count();
		console.log('Conversation count in sidebar:', count);

		printLogs(logCapture);

		expect(count).toBeGreaterThanOrEqual(2);
	});

	// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
	test.skip('should navigate between conversations and show messages', async ({ page }) => {
		test.slow(); // Double timeout for multi-conversation test (2+ LLM roundtrips)
		const logCapture = setupLogCapture(page);

		// Create two conversations
		await page.goto('/');
		await page.waitForTimeout(1000);

		// First conversation
		await page.locator('textarea').fill('UNIQUE_MESSAGE_A_12345');
		await page.getByTestId('send-button').click();
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });
		const urlA = page.url();

		// Wait for response
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 30000 });

		await page.screenshot({ path: 'test-results/screenshots/api-flow-10-conv-a.png' });

		// Create second conversation
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		await page.locator('textarea').fill('UNIQUE_MESSAGE_B_67890');
		await page.getByTestId('send-button').click();
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });

		// Wait for response
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 30000 });

		await page.screenshot({ path: 'test-results/screenshots/api-flow-11-conv-b.png' });

		// Now navigate back to conversation A
		// Find the conversation button that doesn't contain message B
		const sidebar = page.locator('aside');

		// Click first conversation (should be A, since sorted by updatedAt desc)
		// Actually we need to find by content - let's check if messages show in sidebar preview
		// For now, just go back via URL
		await page.goto(urlA);

		await page.screenshot({ path: 'test-results/screenshots/api-flow-12-back-to-conv-a.png' });

		// Messages from A should be visible
		const mainContent = page.locator('main');
		const hasMessageA = await mainContent.locator(':has-text("UNIQUE_MESSAGE_A_12345")').count();
		console.log('Has message A after navigation:', hasMessageA);

		printLogs(logCapture);

		expect(hasMessageA).toBeGreaterThan(0);
	});

	// SKIP: SSE streaming unreliable in Docker E2E environment - network errors interrupt long-running streams
	test.skip('should handle clicking conversation in sidebar', async ({ page }) => {
		const logCapture = setupLogCapture(page);

		// Create a conversation
		await page.goto('/');
		await page.waitForTimeout(1000);

		await page.locator('textarea').fill('Sidebar click test message');
		await page.getByTestId('send-button').click();
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 15000 });

		// Wait for response
		await expect(page.getByTestId('send-button')).toBeVisible({ timeout: 30000 });

		const convUrl = page.url();

		await page.screenshot({ path: 'test-results/screenshots/api-flow-13-before-home.png' });

		// Go home
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		await page.screenshot({ path: 'test-results/screenshots/api-flow-14-at-home.png' });

		// Main area should be empty (no messages)
		const mainMessages = page.locator('main .bg-primary, main .bg-muted');
		const messageCountAtHome = await mainMessages.count();
		console.log('Message count at home:', messageCountAtHome);

		// Click the first conversation in sidebar using data-testid
		const convButton = page.getByTestId('conversation-item').first();
		await convButton.click();

		await page.screenshot({ path: 'test-results/screenshots/api-flow-15-after-sidebar-click.png' });

		// Should navigate to the conversation
		await expect(page).toHaveURL(/\/c\/conv_/, { timeout: 5000 });

		// Messages should be visible
		const messageCountAfterClick = await mainMessages.count();
		console.log('Message count after sidebar click:', messageCountAfterClick);

		printLogs(logCapture);

		// Should have messages now
		expect(messageCountAfterClick).toBeGreaterThan(0);
	});
});
