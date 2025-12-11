import { test, expect } from './fixtures';

test.describe('Layout Shell', () => {
	test('renders two-column layout on desktop', async ({ page }) => {
		await page.setViewportSize({ width: 1280, height: 720 });
		await page.goto('/');

		// Screenshot: Full page layout
		await page.screenshot({
			path: 'test-results/screenshots/layout-desktop.png',
			fullPage: true
		});

		// Verify sidebar visible
		const sidebar = page.locator('aside');
		await expect(sidebar).toBeVisible();
		await expect(sidebar).toContainText('Strieber');

		// Verify main content
		const main = page.locator('main');
		await expect(main).toBeVisible();

		// Verify chat input area
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		await expect(textarea).toBeVisible();
		// Send button is next to textarea in the input area
		const sendButton = page.getByTestId('send-button');
		await expect(sendButton).toBeVisible();
	});

	test('shows hamburger menu on mobile with sidebar off-screen', async ({ page }) => {
		await page.setViewportSize({ width: 375, height: 667 });
		await page.goto('/');

		// Screenshot: Mobile layout
		await page.screenshot({
			path: 'test-results/screenshots/layout-mobile.png',
			fullPage: true
		});

		// Hamburger button should be visible
		const hamburger = page.getByTestId('sidebar-toggle');
		await expect(hamburger).toBeVisible();

		// Mobile header should show title
		const header = page.locator('header');
		await expect(header).toContainText('Strieber');

		// Sidebar should be off-screen (not in viewport)
		const sidebar = page.getByTestId('sidebar');
		const box = await sidebar.boundingBox();
		expect(box).not.toBeNull();
		expect(box!.x + box!.width).toBeLessThanOrEqual(0);

		// Chat input should still be visible on mobile
		const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
		await expect(textarea).toBeVisible();
	});
});
