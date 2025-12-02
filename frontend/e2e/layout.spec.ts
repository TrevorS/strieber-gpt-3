import { test, expect } from '@playwright/test';

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
	});

	test('hides sidebar on mobile', async ({ page }) => {
		await page.setViewportSize({ width: 375, height: 667 });
		await page.goto('/');

		// Screenshot: Mobile layout
		await page.screenshot({
			path: 'test-results/screenshots/layout-mobile.png',
			fullPage: true
		});

		// Sidebar should be hidden
		const sidebar = page.locator('aside');
		await expect(sidebar).not.toBeVisible();
	});
});
