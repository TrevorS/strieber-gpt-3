/**
 * Unit tests for localStorage persistence utilities
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { clearStorage, loadConversations, saveConversations } from '../storage';
import { createConversation, createMessage } from '$lib/stores/types';

describe('storage utilities', () => {
	beforeEach(() => {
		localStorage.clear();
	});

	afterEach(() => {
		localStorage.clear();
	});

	describe('saveConversations', () => {
		it('should save conversations with version field', () => {
			const conversations = [createConversation({ title: 'Test Chat' })];
			saveConversations(conversations, null);

			const stored = JSON.parse(localStorage.getItem('strieber-conversations')!);
			expect(stored.version).toBe(2);
			expect(stored.conversations).toHaveLength(1);
			expect(stored.conversations[0].title).toBe('Test Chat');
		});

		it('should save activeId', () => {
			const conversations = [createConversation({ id: 'test-id' })];
			saveConversations(conversations, 'test-id');

			const stored = JSON.parse(localStorage.getItem('strieber-conversations')!);
			expect(stored.activeId).toBe('test-id');
		});

		it('should save null activeId', () => {
			saveConversations([], null);

			const stored = JSON.parse(localStorage.getItem('strieber-conversations')!);
			expect(stored.activeId).toBeNull();
		});

		it('should handle empty conversations array', () => {
			saveConversations([], null);

			const stored = JSON.parse(localStorage.getItem('strieber-conversations')!);
			expect(stored.conversations).toEqual([]);
		});

		it('should not throw on localStorage errors', () => {
			const mockSetItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
				throw new Error('QuotaExceededError');
			});

			expect(() => saveConversations([], null)).not.toThrow();
			mockSetItem.mockRestore();
		});
	});

	describe('loadConversations', () => {
		it('should load saved conversations', () => {
			const conversations = [createConversation({ title: 'Loaded Chat' })];
			saveConversations(conversations, conversations[0].id);

			const result = loadConversations();
			expect(result).not.toBeNull();
			expect(result?.conversations).toHaveLength(1);
			expect(result?.conversations[0].title).toBe('Loaded Chat');
			expect(result?.activeId).toBe(conversations[0].id);
		});

		it('should return null when no data exists', () => {
			expect(loadConversations()).toBeNull();
		});

		it('should return null for corrupt JSON', () => {
			localStorage.setItem('strieber-conversations', 'not valid json {{{');
			expect(loadConversations()).toBeNull();
		});

		it('should return null for invalid schema (missing version)', () => {
			localStorage.setItem(
				'strieber-conversations',
				JSON.stringify({ conversations: [], activeId: null })
			);
			expect(loadConversations()).toBeNull();
		});

		it('should return null for invalid schema (missing conversations)', () => {
			localStorage.setItem(
				'strieber-conversations',
				JSON.stringify({ version: 1, activeId: null })
			);
			expect(loadConversations()).toBeNull();
		});

		it('should return null for invalid schema (conversations not array)', () => {
			localStorage.setItem(
				'strieber-conversations',
				JSON.stringify({ version: 1, conversations: 'not an array', activeId: null })
			);
			expect(loadConversations()).toBeNull();
		});

		it('should preserve conversation message arrays', () => {
			const conv = createConversation();
			conv.messages = [createMessage('user', 'Hello'), createMessage('assistant', 'Hi there!')];
			saveConversations([conv], conv.id);

			const result = loadConversations();
			expect(result?.conversations[0].messages).toHaveLength(2);
			expect(result?.conversations[0].messages[0].role).toBe('user');
			expect(result?.conversations[0].messages[0].content).toBe('Hello');
			expect(result?.conversations[0].messages[1].role).toBe('assistant');
			expect(result?.conversations[0].messages[1].content).toBe('Hi there!');
		});

		it('should preserve serverConversationId', () => {
			const conv = createConversation();
			conv.serverConversationId = 'conv_abc123';
			saveConversations([conv], conv.id);

			const result = loadConversations();
			expect(result?.conversations[0].serverConversationId).toBe('conv_abc123');
		});

		it('should preserve timestamps', () => {
			const now = Date.now();
			const conv = createConversation({ createdAt: now, updatedAt: now + 1000 });
			saveConversations([conv], conv.id);

			const result = loadConversations();
			expect(result?.conversations[0].createdAt).toBe(now);
			expect(result?.conversations[0].updatedAt).toBe(now + 1000);
		});
	});

	describe('clearStorage', () => {
		it('should remove stored data', () => {
			saveConversations([createConversation()], null);
			expect(localStorage.getItem('strieber-conversations')).not.toBeNull();

			clearStorage();
			expect(localStorage.getItem('strieber-conversations')).toBeNull();
		});

		it('should not throw when data does not exist', () => {
			expect(() => clearStorage()).not.toThrow();
		});

		it('should not throw on localStorage errors', () => {
			const mockRemoveItem = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
				throw new Error('Storage error');
			});

			expect(() => clearStorage()).not.toThrow();
			mockRemoveItem.mockRestore();
		});
	});
});
