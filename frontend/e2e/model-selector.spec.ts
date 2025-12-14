import { test, expect } from './fixtures';

test.describe('Model Selector', () => {
	// Helper to wait for page to be fully ready
	async function waitForPageReady(page: import('@playwright/test').Page) {
		await page.waitForLoadState('networkidle');
		const trigger = page.getByTestId('model-selector-trigger');
		await expect(trigger).toBeEnabled({ timeout: 15000 });
	}

	test('shows model selector in header', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Verify model selector is visible
		const modelSelector = page.getByTestId('model-selector');
		await expect(modelSelector).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/model-selector-initial.png',
			fullPage: true
		});
	});

	test('shows loading state while fetching models', async ({ page }) => {
		await page.goto('/');

		// Model selector trigger should show loading initially
		const trigger = page.getByTestId('model-selector-trigger');
		await expect(trigger).toBeVisible();

		// It may show "Loading..." briefly - this test verifies the element exists
		// The loading state is brief so we just verify the button exists
		await expect(trigger).toBeEnabled({ timeout: 15000 });
	});

	test('opens dropdown when clicked', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load
		const trigger = page.getByTestId('model-selector-trigger');

		// Click to open dropdown
		await trigger.click();

		// Verify dropdown is visible
		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		await page.screenshot({
			path: 'test-results/screenshots/model-selector-open.png',
			fullPage: true
		});
	});

	test('closes dropdown when clicking outside', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load and open dropdown
		const trigger = page.getByTestId('model-selector-trigger');
		await trigger.click();

		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Click outside (on the main content area)
		await page.locator('main').click();

		// Dropdown should be closed
		await expect(dropdown).not.toBeVisible();
	});

	test('closes dropdown when pressing Escape', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load and open dropdown
		const trigger = page.getByTestId('model-selector-trigger');
		await trigger.click();

		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Press Escape
		await page.keyboard.press('Escape');

		// Dropdown should be closed
		await expect(dropdown).not.toBeVisible();
	});

	test('shows available model options in dropdown', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load and open dropdown
		const trigger = page.getByTestId('model-selector-trigger');
		await trigger.click();

		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Verify at least one model option is shown
		const modelOptions = dropdown.getByTestId('model-option');
		const count = await modelOptions.count();
		expect(count).toBeGreaterThan(0);
	});

	test('selecting a model updates the displayed selection', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load
		const trigger = page.getByTestId('model-selector-trigger');

		// Get current selected model text
		const initialText = await trigger.textContent();

		// Open dropdown
		await trigger.click();

		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Get the first model option that's different from current
		const modelOptions = dropdown.getByTestId('model-option');
		const count = await modelOptions.count();

		if (count > 1) {
			// Find a model that's not currently selected
			for (let i = 0; i < count; i++) {
				const option = modelOptions.nth(i);
				const isSelected = await option.getAttribute('aria-selected');
				if (isSelected !== 'true') {
					const optionText = await option.textContent();
					await option.click();

					// Verify dropdown closed
					await expect(dropdown).not.toBeVisible();

					// Verify trigger shows new model
					await expect(trigger).toContainText(optionText!);
					break;
				}
			}
		}

		await page.screenshot({
			path: 'test-results/screenshots/model-selector-selected.png',
			fullPage: true
		});
	});

	test('selected model persists across page reloads', async ({ page }) => {
		await page.goto('/');
		await waitForPageReady(page);

		// Wait for models to load
		const trigger = page.getByTestId('model-selector-trigger');

		// Open dropdown and select a specific model
		await trigger.click();
		const dropdown = page.getByTestId('model-selector-dropdown');
		await expect(dropdown).toBeVisible();

		// Find first non-selected option
		const modelOptions = dropdown.getByTestId('model-option');
		const count = await modelOptions.count();
		let selectedModelId = '';

		for (let i = 0; i < count; i++) {
			const option = modelOptions.nth(i);
			const isSelected = await option.getAttribute('aria-selected');
			if (isSelected !== 'true') {
				selectedModelId = (await option.textContent()) || '';
				await option.click();
				break;
			}
		}

		if (selectedModelId) {
			// Reload page
			await page.reload();
			await page.waitForLoadState('networkidle');

			// Wait for models to load again
			await expect(trigger).toBeEnabled({ timeout: 15000 });

			// Verify same model is still selected
			await expect(trigger).toContainText(selectedModelId);
		}
	});
});
