/**
 * localStorage persistence for conversations
 *
 * Handles saving and loading conversation state with version tracking
 * for future schema migrations.
 */
import type { Conversation } from '$lib/stores/types';

const STORAGE_KEY = 'strieber-conversations';
const CURRENT_VERSION = 2;

/**
 * Schema for stored data
 */
interface StorageSchema {
	version: number;
	conversations: Conversation[];
	activeId: string | null;
}

/**
 * Save conversations to localStorage
 *
 * @param conversations - Array of conversations to save
 * @param activeId - ID of the currently active conversation
 */
export function saveConversations(conversations: Conversation[], activeId: string | null): void {
	try {
		const data: StorageSchema = {
			version: CURRENT_VERSION,
			conversations,
			activeId
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	} catch (error) {
		console.warn('Failed to save conversations:', error);
	}
}

/**
 * Load conversations from localStorage
 *
 * @returns Loaded data or null if not found/invalid
 */
export function loadConversations(): {
	conversations: Conversation[];
	activeId: string | null;
} | null {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return null;

		const data = JSON.parse(raw) as StorageSchema;

		// Validate schema structure
		if (typeof data.version !== 'number' || !Array.isArray(data.conversations)) {
			console.warn('Invalid storage schema, resetting');
			return null;
		}

		// Version check (migration point for future versions)
		if (data.version !== CURRENT_VERSION) {
			console.warn(`Storage version mismatch: ${data.version} vs ${CURRENT_VERSION}`);
			return null;
		}

		return {
			conversations: data.conversations,
			activeId: data.activeId
		};
	} catch (error) {
		console.warn('Failed to load conversations:', error);
		return null;
	}
}

/**
 * Clear all stored conversation data
 */
export function clearStorage(): void {
	try {
		localStorage.removeItem(STORAGE_KEY);
	} catch (error) {
		console.warn('Failed to clear storage:', error);
	}
}
