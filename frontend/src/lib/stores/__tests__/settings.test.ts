/**
 * Unit tests for settings store
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('settingsStore', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		localStorage.clear();
		// Reset module to get fresh store instance
		vi.resetModules();
	});

	afterEach(() => {
		vi.restoreAllMocks();
		localStorage.clear();
	});

	it('should have default values', async () => {
		const { settingsStore } = await import('../settings.svelte');

		expect(settingsStore.selectedModel).toBe('gpt-oss-120b');
		expect(settingsStore.temperature).toBe(1.0);
		expect(settingsStore.theme).toBe('system');
	});

	it('should load saved settings from localStorage', async () => {
		localStorage.setItem(
			'strieber-settings',
			JSON.stringify({
				selectedModel: 'custom-model',
				temperature: 0.7,
				theme: 'dark'
			})
		);

		const { settingsStore } = await import('../settings.svelte');

		expect(settingsStore.selectedModel).toBe('custom-model');
		expect(settingsStore.temperature).toBe(0.7);
		expect(settingsStore.theme).toBe('dark');
	});

	it('should persist model changes', async () => {
		const { settingsStore } = await import('../settings.svelte');

		settingsStore.setModel('new-model');

		expect(settingsStore.selectedModel).toBe('new-model');
		const stored = JSON.parse(localStorage.getItem('strieber-settings')!);
		expect(stored.selectedModel).toBe('new-model');
	});

	it('should persist temperature changes', async () => {
		const { settingsStore } = await import('../settings.svelte');

		settingsStore.setTemperature(1.5);

		expect(settingsStore.temperature).toBe(1.5);
		const stored = JSON.parse(localStorage.getItem('strieber-settings')!);
		expect(stored.temperature).toBe(1.5);
	});

	it('should clamp temperature between 0 and 2', async () => {
		const { settingsStore } = await import('../settings.svelte');

		settingsStore.setTemperature(-1);
		expect(settingsStore.temperature).toBe(0);

		settingsStore.setTemperature(5);
		expect(settingsStore.temperature).toBe(2);
	});

	it('should persist theme changes', async () => {
		const { settingsStore } = await import('../settings.svelte');

		settingsStore.setTheme('light');

		expect(settingsStore.theme).toBe('light');
		const stored = JSON.parse(localStorage.getItem('strieber-settings')!);
		expect(stored.theme).toBe('light');
	});

	it('should merge saved settings with defaults for missing keys', async () => {
		// Simulate old settings without theme
		localStorage.setItem(
			'strieber-settings',
			JSON.stringify({
				selectedModel: 'old-model'
			})
		);

		const { settingsStore } = await import('../settings.svelte');

		expect(settingsStore.selectedModel).toBe('old-model');
		expect(settingsStore.temperature).toBe(1.0); // default
		expect(settingsStore.theme).toBe('system'); // default
	});
});
