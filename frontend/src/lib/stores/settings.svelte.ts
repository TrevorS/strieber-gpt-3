/**
 * Settings Store
 *
 * Svelte 5 runes-based store for user preferences.
 * Persisted to localStorage.
 */

import { browser } from '$app/environment';
import type { Model } from '$lib/api/models';

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

	/** Available models from API */
	models = $state<Model[]>([]);

	/** Whether the currently selected model supports vision/image inputs */
	supportsVision = $derived(() => {
		const model = this.models.find((m) => m.id === this.selectedModel);
		return model?.supports_vision ?? false;
	});

	/** Which tools the currently selected model supports (null = all, [] = none) */
	supportedTools = $derived(() => {
		const model = this.models.find((m) => m.id === this.selectedModel);
		return model?.supported_tools ?? null;
	});

	constructor() {
		if (browser) {
			const saved = loadSettings();
			this.selectedModel = saved.selectedModel;
			this.temperature = saved.temperature;
			this.theme = saved.theme;
		}
	}

	/**
	 * Set available models (called after fetching from API).
	 */
	setModels(models: Model[]): void {
		this.models = models;
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
	 * Filter tools based on current model's supported_tools.
	 * Returns empty array if model supports no tools.
	 * Returns all tools if model supports all tools (null).
	 */
	filterTools<T extends { type: string }>(tools: T[]): T[] {
		const supported = this.supportedTools();
		// null = all tools supported
		if (supported === null) {
			return tools;
		}
		// Filter to only supported tool types
		return tools.filter((tool) => supported.includes(tool.type));
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
