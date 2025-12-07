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
	systemPrompt: string;
	enabledTools: Record<string, boolean>;
	sidebarCollapsed: boolean;
}

const defaultSettings: SettingsData = {
	selectedModel: DEFAULT_MODEL,
	temperature: 1.0,
	theme: 'system',
	systemPrompt: '',
	enabledTools: {
		web_search: true,
		code_interpreter: true,
		weather: true,
		reader: true,
		zimage_turbo: true
	},
	sidebarCollapsed: false
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

	/** Custom system prompt */
	systemPrompt = $state('');

	/** Tool enable/disable state */
	enabledTools = $state<Record<string, boolean>>({
		web_search: true,
		code_interpreter: true,
		weather: true,
		reader: true,
		zimage_turbo: true
	});

	/** Whether sidebar is collapsed (desktop only) */
	sidebarCollapsed = $state(false);

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
			this.systemPrompt = saved.systemPrompt;
			this.enabledTools = saved.enabledTools;
			this.sidebarCollapsed = saved.sidebarCollapsed ?? false;
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
	 * Update system prompt and persist.
	 */
	setSystemPrompt(prompt: string): void {
		this.systemPrompt = prompt;
		this.persist();
	}

	/**
	 * Enable or disable a specific tool.
	 */
	setToolEnabled(toolId: string, enabled: boolean): void {
		this.enabledTools = { ...this.enabledTools, [toolId]: enabled };
		this.persist();
	}

	/**
	 * Enable or disable all tools at once.
	 */
	setAllToolsEnabled(enabled: boolean): void {
		const newState: Record<string, boolean> = {};
		for (const toolId of Object.keys(this.enabledTools)) {
			newState[toolId] = enabled;
		}
		this.enabledTools = newState;
		this.persist();
	}

	/**
	 * Toggle sidebar collapsed state and persist.
	 */
	toggleSidebarCollapsed(): void {
		this.sidebarCollapsed = !this.sidebarCollapsed;
		this.persist();
	}

	/**
	 * Filter tools based on current model's supported_tools AND user's enabled tools.
	 * Returns empty array if model supports no tools.
	 * Returns filtered tools based on model support and user preferences.
	 */
	filterTools<T extends { type: string }>(tools: T[]): T[] {
		const supported = this.supportedTools();

		return tools.filter((tool) => {
			// Check model support (null = all supported)
			if (supported !== null && !supported.includes(tool.type)) {
				return false;
			}
			// Check user's enabled state (default to true if not in map)
			return this.enabledTools[tool.type] !== false;
		});
	}

	/**
	 * Persist current settings to localStorage.
	 */
	private persist(): void {
		saveSettings({
			selectedModel: this.selectedModel,
			temperature: this.temperature,
			theme: this.theme,
			systemPrompt: this.systemPrompt,
			enabledTools: this.enabledTools,
			sidebarCollapsed: this.sidebarCollapsed
		});
	}
}

export const settingsStore = new SettingsStore();
