/**
 * Settings Store
 *
 * Svelte 5 runes-based store for user preferences.
 * Persisted to localStorage.
 */

import { browser } from '$app/environment';

const STORAGE_KEY = 'strieber-settings';
const DEFAULT_MODEL = 'gpt-oss-120b';

interface SettingsData {
	selectedModel: string;
	temperature: number;
	theme: 'light' | 'dark' | 'system';
}

const defaultSettings: SettingsData = {
	selectedModel: DEFAULT_MODEL,
	temperature: 1.0,
	theme: 'system'
};

/**
 * Load settings from localStorage.
 */
function loadSettings(): SettingsData {
	if (!browser) return defaultSettings;

	try {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (stored) {
			const parsed = JSON.parse(stored);
			// Merge with defaults to handle new fields
			return { ...defaultSettings, ...parsed };
		}
	} catch {
		console.warn('Failed to load settings from localStorage');
	}

	return defaultSettings;
}

/**
 * Save settings to localStorage.
 */
function saveSettings(settings: SettingsData): void {
	if (!browser) return;

	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
	} catch {
		console.warn('Failed to save settings to localStorage');
	}
}

/**
 * Settings store class using Svelte 5 runes.
 */
class SettingsStore {
	/** Currently selected model ID */
	selectedModel = $state(DEFAULT_MODEL);

	/** Temperature setting (0.0 - 2.0) */
	temperature = $state(1.0);

	/** Theme preference */
	theme = $state<'light' | 'dark' | 'system'>('system');

	constructor() {
		if (browser) {
			const saved = loadSettings();
			this.selectedModel = saved.selectedModel;
			this.temperature = saved.temperature;
			this.theme = saved.theme;
		}
	}

	/**
	 * Update selected model and persist.
	 */
	setModel(modelId: string): void {
		this.selectedModel = modelId;
		this.persist();
	}

	/**
	 * Update temperature and persist.
	 */
	setTemperature(value: number): void {
		this.temperature = Math.max(0, Math.min(2, value));
		this.persist();
	}

	/**
	 * Update theme and persist.
	 */
	setTheme(value: 'light' | 'dark' | 'system'): void {
		this.theme = value;
		this.persist();
	}

	/**
	 * Persist current settings to localStorage.
	 */
	private persist(): void {
		saveSettings({
			selectedModel: this.selectedModel,
			temperature: this.temperature,
			theme: this.theme
		});
	}
}

export const settingsStore = new SettingsStore();
