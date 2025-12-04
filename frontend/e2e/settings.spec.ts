import { test, expect } from '@playwright/test';

test.describe('Settings Panel', () => {
	test('opens settings panel when clicking settings button', async ({ page }) => {
		await page.goto('/');

		// Wait for the page to be fully loaded (hydration complete)
		await page.waitForLoadState('networkidle');

		// Wait for the settings button to be ready and enabled
		const settingsButton = page.getByTestId('settings-button');
		await expect(settingsButton).toBeVisible({ timeout: 10000 });
		await expect(settingsButton).toBeEnabled();

		// Small delay to ensure hydration is complete
		await page.waitForTimeout(200);

		// Click settings button
		await settingsButton.click();

		// Verify settings panel is visible
		const settingsPanel = page.getByTestId('settings-panel');
		await expect(settingsPanel).toBeVisible();
		await expect(settingsPanel).toContainText('Settings');

		await page.screenshot({
			path: 'test-results/screenshots/settings-open.png',
			fullPage: true
		});
	});

	test('closes settings panel when clicking backdrop', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();
		await expect(page.getByTestId('settings-panel')).toBeVisible();

		// Click backdrop to close
		await page.getByTestId('settings-backdrop').click();

		// Verify panel is closed
		await expect(page.getByTestId('settings-panel')).not.toBeVisible();
	});

	test('closes settings panel when pressing Escape', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();
		await expect(page.getByTestId('settings-panel')).toBeVisible();

		// Press Escape
		await page.keyboard.press('Escape');

		// Verify panel is closed
		await expect(page.getByTestId('settings-panel')).not.toBeVisible();
	});

	test('closes settings panel when clicking X button', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();
		const panel = page.getByTestId('settings-panel');
		await expect(panel).toBeVisible();

		// Click close button (X icon)
		await panel.getByRole('button', { name: 'Close settings' }).click();

		// Verify panel is closed
		await expect(panel).not.toBeVisible();
	});

	test('shows theme toggle with all options', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();

		// Verify theme toggle is visible with all options
		const themeToggle = page.getByTestId('theme-toggle');
		await expect(themeToggle).toBeVisible();

		await expect(page.getByTestId('theme-option-light')).toBeVisible();
		await expect(page.getByTestId('theme-option-dark')).toBeVisible();
		await expect(page.getByTestId('theme-option-system')).toBeVisible();
	});

	test('switching to dark theme applies dark class to document', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();

		// Click dark theme button
		await page.getByTestId('theme-option-dark').click();

		// Verify dark class is applied to html element
		const isDark = await page.evaluate(() => {
			return document.documentElement.classList.contains('dark');
		});
		expect(isDark).toBe(true);

		await page.screenshot({
			path: 'test-results/screenshots/settings-dark-theme.png',
			fullPage: true
		});
	});

	test('switching to light theme removes dark class from document', async ({ page }) => {
		await page.goto('/');

		// First switch to dark
		await page.getByTestId('settings-button').click();
		await page.getByTestId('theme-option-dark').click();

		// Then switch to light
		await page.getByTestId('theme-option-light').click();

		// Verify dark class is removed
		const isDark = await page.evaluate(() => {
			return document.documentElement.classList.contains('dark');
		});
		expect(isDark).toBe(false);

		await page.screenshot({
			path: 'test-results/screenshots/settings-light-theme.png',
			fullPage: true
		});
	});

	test('shows temperature slider with current value', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();

		// Verify temperature slider is visible
		const temperatureSlider = page.getByTestId('temperature-slider');
		await expect(temperatureSlider).toBeVisible();
		await expect(temperatureSlider).toContainText('Temperature');
		await expect(temperatureSlider).toContainText('Precise');
		await expect(temperatureSlider).toContainText('Creative');
	});

	test('adjusting temperature slider updates displayed value', async ({ page }) => {
		await page.goto('/');

		// Open settings
		await page.getByTestId('settings-button').click();

		const slider = page.locator('#temperature');
		const temperatureSlider = page.getByTestId('temperature-slider');

		// Get initial value
		const initialValue = await slider.inputValue();

		// Move slider to a different value (0.5)
		await slider.fill('0.5');

		// Verify displayed value changed
		await expect(temperatureSlider).toContainText('0.5');

		// Move to max value
		await slider.fill('2');
		await expect(temperatureSlider).toContainText('2.0');

		await page.screenshot({
			path: 'test-results/screenshots/settings-temperature.png',
			fullPage: true
		});
	});

	test('settings persist across page reloads', async ({ page }) => {
		await page.goto('/');

		// Open settings and change theme to dark
		await page.getByTestId('settings-button').click();
		await page.getByTestId('theme-option-dark').click();

		// Reload page
		await page.reload();

		// Verify dark theme is still applied
		const isDark = await page.evaluate(() => {
			return document.documentElement.classList.contains('dark');
		});
		expect(isDark).toBe(true);

		// Open settings and verify dark is selected
		await page.getByTestId('settings-button').click();
		const darkButton = page.getByTestId('theme-option-dark');
		await expect(darkButton).toHaveAttribute('aria-pressed', 'true');
	});
});
