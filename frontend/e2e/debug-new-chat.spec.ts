import { test, expect } from '@playwright/test';
import { setupLogCapture, filterLogs } from './helpers/logger';

test.describe('Debug: New Chat Flow', () => {
	test.setTimeout(90000);

	test('MOBILE: trace New Chat via hamburger menu', async ({ page }) => {
		// Set mobile viewport
		await page.setViewportSize({ width: 375, height: 667 });

		const logs = setupLogCapture(page);

		await page.goto('/');

		// Create a conversation
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		const conversationId = page.url().split('/c/')[1];
		console.log('\n=== MOBILE SETUP COMPLETE ===');
		console.log('Conversation ID:', conversationId);

		// Clear logs
		logs.structuredLogs.length = 0;

		// Open sidebar via hamburger
		console.log('\n=== OPENING SIDEBAR ===');
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();

		// Click New Chat in sidebar
		console.log('\n=== CLICKING NEW CHAT (MOBILE) ===');
		await sidebar.getByRole('button', { name: 'New Chat' }).first().click();

		// Wait for effects
		await page.waitForTimeout(1000);

		// Print debug logs
		console.log('\n=== MOBILE DEBUG LOG TRACE ===');
		const debugLogs = logs.structuredLogs.filter(
			(log) => log.message.startsWith('[') || log.category === 'debug'
		);
		for (const log of debugLogs) {
			console.log(`[${log.level}] ${log.message}`, JSON.stringify(log.data, null, 2));
		}
		console.log('=== END TRACE ===\n');

		// Check results
		const finalUrl = page.url();
		console.log('Final URL:', finalUrl);

		const userMessages = page.locator('div.bg-primary');
		const messageCount = await userMessages.count();
		console.log('Message count:', messageCount);

		// Look for the bug
		const settingBackLogs = debugLogs.filter((log) =>
			log.message.includes('SETTING ACTIVE BACK')
		);
		if (settingBackLogs.length > 0) {
			console.log('\n!!! MOBILE BUG FOUND !!!');
			for (const log of settingBackLogs) {
				console.log(JSON.stringify(log.data, null, 2));
			}
		}

		await page.screenshot({
			path: 'test-results/screenshots/debug-mobile-new-chat.png',
			fullPage: true
		});

		await expect(page).toHaveURL('/');
		await expect(userMessages).toHaveCount(0, { timeout: 1000 });
	});

	test('trace New Chat click flow and identify race condition', async ({ page }) => {
		const logs = setupLogCapture(page);

		await page.goto('/');

		// Create a conversation first
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		const conversationUrl = page.url();
		const conversationId = conversationUrl.split('/c/')[1];
		console.log('\n=== SETUP COMPLETE ===');
		console.log('Created conversation:', conversationId);
		console.log('URL:', conversationUrl);

		// Clear log buffer to focus on New Chat flow
		logs.structuredLogs.length = 0;

		// Take screenshot before clicking New Chat
		await page.screenshot({
			path: 'test-results/screenshots/debug-before-new-chat.png',
			fullPage: true
		});

		// Click New Chat button
		console.log('\n=== CLICKING NEW CHAT ===');
		const newChatButton = page.getByRole('button', { name: 'New Chat' }).first();
		await newChatButton.click();

		// Wait for effects to settle
		await page.waitForTimeout(1000);

		// Print all debug logs in chronological order
		console.log('\n=== DEBUG LOG TRACE (chronological) ===');
		const debugLogs = logs.structuredLogs.filter(
			(log) => log.message.startsWith('[') || log.category === 'debug'
		);
		for (const log of debugLogs) {
			console.log(`[${log.level}] ${log.message}`, JSON.stringify(log.data, null, 2));
		}
		console.log('=== END TRACE ===\n');

		// Check final URL
		const finalUrl = page.url();
		console.log('Final URL:', finalUrl);

		// Check message count
		const userMessages = page.locator('div.bg-primary');
		const messageCount = await userMessages.count();
		console.log('Message count on screen:', messageCount);

		// Take screenshot after
		await page.screenshot({
			path: 'test-results/screenshots/debug-after-new-chat.png',
			fullPage: true
		});

		// Look for the smoking gun: [ConvPage] Effect: SETTING ACTIVE BACK
		const settingBackLogs = debugLogs.filter((log) =>
			log.message.includes('SETTING ACTIVE BACK')
		);

		if (settingBackLogs.length > 0) {
			console.log('\n!!! FOUND THE BUG !!!');
			console.log('The conversation page effect reset activeId:');
			for (const log of settingBackLogs) {
				console.log(JSON.stringify(log.data, null, 2));
			}
		}

		// Assertions
		await expect(page).toHaveURL('/');

		// This assertion will fail if the bug is present
		await expect(userMessages).toHaveCount(0, {
			timeout: 1000 // Short timeout since we already waited
		});
	});

	test('verify second click works (control test)', async ({ page }) => {
		const logs = setupLogCapture(page);

		await page.goto('/');

		// Create a conversation
		const textarea = page.locator('textarea[placeholder="Send a message..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		// First click
		console.log('\n=== FIRST CLICK ===');
		logs.structuredLogs.length = 0;
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await page.waitForTimeout(500);

		const firstClickMessages = await page.locator('div.bg-primary').count();
		console.log('Message count after first click:', firstClickMessages);

		// Second click (should definitely work)
		console.log('\n=== SECOND CLICK ===');
		logs.structuredLogs.length = 0;
		await page.getByRole('button', { name: 'New Chat' }).first().click();
		await page.waitForTimeout(500);

		const secondClickMessages = await page.locator('div.bg-primary').count();
		console.log('Message count after second click:', secondClickMessages);

		// Take screenshot
		await page.screenshot({
			path: 'test-results/screenshots/debug-second-click.png',
			fullPage: true
		});

		// Second click should always work
		expect(secondClickMessages).toBe(0);
	});
});
