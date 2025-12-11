import { expect, Page } from '@playwright/test';

/**
 * Default timeout for waiting for LLM streaming to complete.
 * Set high to account for slow local LLM responses.
 */
export const STREAMING_TIMEOUT = 60000;

/**
 * Extended timeout for multi-turn conversations or complex operations.
 */
export const EXTENDED_TIMEOUT = 90000;

/**
 * Wait for streaming to complete by checking send button is visible and enabled.
 * Uses longer timeout for slow LLM responses.
 */
export async function waitForStreamingComplete(page: Page, timeout = STREAMING_TIMEOUT) {
	const sendButton = page.getByTestId('send-button');
	await expect(sendButton).toBeVisible({ timeout });
	await expect(sendButton).toBeEnabled({ timeout: 5000 });
}

/**
 * Send a message and wait for streaming to complete.
 * Handles the full flow: fill textarea, click send, wait for URL navigation,
 * and wait for streaming to finish.
 */
export async function sendMessageAndWait(
	page: Page,
	message: string,
	options: { timeout?: number; waitForUrl?: boolean } = {}
) {
	const { timeout = STREAMING_TIMEOUT, waitForUrl = true } = options;

	const textarea = page.locator('textarea[placeholder="Message Strieber GPT..."]');
	const sendButton = page.getByTestId('send-button');

	await textarea.fill(message);
	await sendButton.click();

	if (waitForUrl) {
		await expect(page).toHaveURL(/\/c\/.+/, { timeout: 15000 });
	}

	await waitForStreamingComplete(page, timeout);
}
