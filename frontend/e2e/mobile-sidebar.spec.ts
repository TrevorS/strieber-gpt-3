import { test, expect } from '@playwright/test';

test.describe('Mobile Sidebar', () => {
	test.setTimeout(60000);

	test.beforeEach(async ({ page }) => {
		// Set mobile viewport
		await page.setViewportSize({ width: 375, height: 667 });
	});

	test('hamburger menu is visible on mobile, hidden on desktop', async ({ page }) => {
		await page.goto('/');

		// Hamburger should be visible on mobile
		const hamburger = page.getByTestId('sidebar-toggle');
		await expect(hamburger).toBeVisible();

		// Sidebar should be off-screen (not in viewport)
		const sidebar = page.getByTestId('sidebar');
		const box = await sidebar.boundingBox();
		expect(box).not.toBeNull();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0); // Sidebar is off left edge

		await page.screenshot({
			path: 'test-results/screenshots/mobile-sidebar-closed.png',
			fullPage: true
		});

		// Switch to desktop
		await page.setViewportSize({ width: 1280, height: 720 });

		// Hamburger should be hidden
		await expect(hamburger).not.toBeVisible();

		// Sidebar should be visible and in viewport
		await expect(sidebar).toBeInViewport();

		await page.screenshot({
			path: 'test-results/screenshots/desktop-sidebar.png',
			fullPage: true
		});
	});

	test('clicking hamburger opens sidebar with backdrop', async ({ page }) => {
		await page.goto('/');

		const hamburger = page.getByTestId('sidebar-toggle');
		await hamburger.click();

		// Sidebar should be visible (translated to 0)
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toHaveCSS('transform', 'none');

		// Backdrop should appear
		const backdrop = page.getByTestId('sidebar-backdrop');
		await expect(backdrop).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/mobile-sidebar-open.png',
			fullPage: true
		});
	});

	test('clicking backdrop closes sidebar', async ({ page }) => {
		await page.goto('/');

		// Open sidebar
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();

		// Wait for sidebar to be fully open
		await page.waitForTimeout(300);

		// Click backdrop to the RIGHT of the sidebar (sidebar is 256px wide)
		// Click at x=300 to ensure we're clicking on backdrop, not sidebar
		await page.mouse.click(300, 300);

		// Wait for transition
		await page.waitForTimeout(350);

		// Sidebar should be closed (off viewport)
		const box = await sidebar.boundingBox();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0);

		// Backdrop should be gone
		await expect(page.getByTestId('sidebar-backdrop')).not.toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/mobile-sidebar-backdrop-close.png',
			fullPage: true
		});
	});

	test('escape key closes sidebar', async ({ page }) => {
		await page.goto('/');

		// Open sidebar
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();

		// Press Escape
		await page.keyboard.press('Escape');

		// Wait for transition
		await page.waitForTimeout(350);

		// Sidebar should be closed (off viewport)
		const box = await sidebar.boundingBox();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0);
	});

	test('New Chat button closes sidebar', async ({ page }) => {
		await page.goto('/');

		// Open sidebar
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();

		// Click New Chat button inside sidebar
		await sidebar.getByRole('button', { name: 'New Chat' }).click();

		// Wait for transition
		await page.waitForTimeout(350);

		// Sidebar should be closed (off viewport)
		const box = await sidebar.boundingBox();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0);

		// Should be on home page
		await expect(page).toHaveURL('/');
	});

	test('New Chat clears conversation and allows new message on mobile', async ({ page }) => {
		await page.goto('/');

		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');
		const sidebar = page.getByTestId('sidebar');

		// Create first conversation
		await textarea.fill('Say "first" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });
		const firstUrl = page.url();

		// Open sidebar and click New Chat (header button, not conversation item)
		await page.getByTestId('sidebar-toggle').click();
		await expect(sidebar).toBeInViewport();
		await sidebar.getByRole('button', { name: 'New Chat' }).first().click();

		// Wait for sidebar transition and navigation
		await page.waitForTimeout(350);
		await expect(page).toHaveURL('/');

		// Verify messages are cleared - NO waiting, test immediately
		const userMessages = page.locator('div.bg-primary');
		const assistantMessages = page.locator('.bg-muted');
		await expect(userMessages).toHaveCount(0);
		await expect(assistantMessages).toHaveCount(0);

		// Immediately send a new message
		await textarea.fill('Say "second" only.');
		await sendButton.click();

		// Should navigate to a NEW conversation URL
		await expect(page).toHaveURL(/\/c\/.+/);
		const secondUrl = page.url();
		expect(secondUrl).not.toBe(firstUrl);

		// Should only see the new message
		await expect(userMessages).toHaveCount(1);
		await expect(userMessages.first()).toContainText('second');

		await page.screenshot({
			path: 'test-results/screenshots/mobile-new-chat-clears.png',
			fullPage: true
		});
	});

	test('selecting conversation closes sidebar and navigates', async ({ page }) => {
		await page.goto('/');

		// First create a conversation
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		const sendButton = page.getByTestId('send-button');

		await textarea.fill('Say "test" only.');
		await sendButton.click();
		await expect(page).toHaveURL(/\/c\/.+/);
		await expect(textarea).toBeEnabled({ timeout: 30000 });

		const conversationUrl = page.url();

		// Go to new chat via hamburger menu
		await page.getByTestId('sidebar-toggle').click();
		const sidebar = page.getByTestId('sidebar');
		await expect(sidebar).toBeInViewport();
		// Use first() to get the New Chat button in the header, not the conversation item
		await sidebar.getByRole('button', { name: 'New Chat' }).first().click();
		await expect(page).toHaveURL('/');

		// Wait for transition
		await page.waitForTimeout(350);

		// Open sidebar and select the conversation
		await page.getByTestId('sidebar-toggle').click();
		await expect(sidebar).toBeInViewport();

		// Click on the conversation item (not the header's New Chat button)
		// Conversation items have a delete button inside them, so we can use that to identify them
		const conversationItem = sidebar.locator('button:has([data-testid="delete-button"])').first();
		await conversationItem.click();

		// Wait for transition
		await page.waitForTimeout(350);

		// Should navigate back to conversation and close sidebar
		await expect(page).toHaveURL(conversationUrl);
		const box = await sidebar.boundingBox();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0);

		await page.screenshot({
			path: 'test-results/screenshots/mobile-sidebar-navigate.png',
			fullPage: true
		});
	});

	test('mobile header shows app title', async ({ page }) => {
		await page.goto('/');

		// Mobile header should show Strieber title
		const header = page.locator('header');
		await expect(header).toContainText('Strieber');
		await expect(header).toBeVisible();
	});

	test('main content has proper padding on mobile', async ({ page }) => {
		await page.goto('/');

		// Main content should have top padding for header
		const main = page.locator('main');
		await expect(main).toHaveCSS('padding-top', '56px'); // pt-14 = 3.5rem = 56px
	});
});
