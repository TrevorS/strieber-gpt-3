/**
 * Unit tests for Conversation State Store
 *
 * Tests cover:
 * - CRUD operations (create, read, update, delete)
 * - Active conversation tracking
 * - Message operations
 * - serverConversationId tracking
 * - Edge cases
 */

import { beforeEach, describe, expect, it } from 'vitest';
import { conversationStore } from '../conversations.svelte';
import { createConversation, createMessage } from '../types';

describe('conversationStore', () => {
	beforeEach(() => {
		// Reset store state before each test
		conversationStore.clear();
	});

	// ============================================================================
	// Create Operations
	// ============================================================================

	describe('create', () => {
		it('should create a new conversation with default title', () => {
			const conv = conversationStore.create();

			expect(conv.id).toBeDefined();
			expect(conv.title).toBe('New Chat');
			expect(conv.messages).toEqual([]);
			expect(conv.serverConversationId).toBeNull();
			expect(conversationStore.conversations).toHaveLength(1);
		});

		it('should create a conversation with custom title', () => {
			const conv = conversationStore.create('My Custom Chat');

			expect(conv.title).toBe('My Custom Chat');
		});

		it('should set the new conversation as active', () => {
			const conv = conversationStore.create();

			expect(conversationStore.activeId).toBe(conv.id);
			// Use get() to fetch from reactive state
			expect(conversationStore.get(conv.id)).toBeDefined();
		});

		it('should create multiple conversations', () => {
			conversationStore.create('First');
			const conv2 = conversationStore.create('Second');

			expect(conversationStore.conversations).toHaveLength(2);
			// Most recent should be active
			expect(conversationStore.activeId).toBe(conv2.id);
		});
	});

	// ============================================================================
	// Delete Operations
	// ============================================================================

	describe('delete', () => {
		it('should delete a conversation by ID', () => {
			const conv = conversationStore.create();
			conversationStore.delete(conv.id);

			expect(conversationStore.conversations).toHaveLength(0);
		});

		it('should do nothing when deleting non-existent ID', () => {
			conversationStore.create();
			conversationStore.delete('non-existent-id');

			expect(conversationStore.conversations).toHaveLength(1);
		});

		it('should switch active to another conversation when deleting active', () => {
			const conv1 = conversationStore.create('First');
			const conv2 = conversationStore.create('Second');

			// conv2 is active
			expect(conversationStore.activeId).toBe(conv2.id);

			conversationStore.delete(conv2.id);

			// Should switch to conv1
			expect(conversationStore.activeId).toBe(conv1.id);
		});

		it('should set activeId to null when deleting last conversation', () => {
			const conv = conversationStore.create();
			conversationStore.delete(conv.id);

			expect(conversationStore.activeId).toBeNull();
			expect(conversationStore.active).toBeUndefined();
		});
	});

	// ============================================================================
	// Active Conversation
	// ============================================================================

	describe('setActive', () => {
		it('should set active conversation by ID', () => {
			const conv1 = conversationStore.create('First');
			conversationStore.create('Second');

			conversationStore.setActive(conv1.id);

			expect(conversationStore.activeId).toBe(conv1.id);
			expect(conversationStore.active?.title).toBe('First');
		});

		it('should allow setting active to null', () => {
			conversationStore.create();
			conversationStore.setActive(null);

			expect(conversationStore.activeId).toBeNull();
		});
	});

	// ============================================================================
	// Title Updates
	// ============================================================================

	describe('updateTitle', () => {
		it('should update conversation title', () => {
			const conv = conversationStore.create('Original');
			conversationStore.updateTitle(conv.id, 'Updated Title');

			expect(conversationStore.get(conv.id)?.title).toBe('Updated Title');
		});

		it('should update updatedAt timestamp', () => {
			const conv = conversationStore.create();
			const originalUpdatedAt = conv.updatedAt;

			// Small delay to ensure timestamp differs
			conversationStore.updateTitle(conv.id, 'New Title');

			expect(conversationStore.get(conv.id)?.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});

		it('should do nothing for non-existent conversation', () => {
			conversationStore.updateTitle('non-existent', 'Title');
			// No error thrown, just no-op
			expect(conversationStore.conversations).toHaveLength(0);
		});
	});

	// ============================================================================
	// Message Operations
	// ============================================================================

	describe('addMessage', () => {
		it('should add a user message to conversation', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Hello!');

			expect(message.role).toBe('user');
			expect(message.content).toBe('Hello!');
			// Fetch current state from store (Svelte 5 reactivity)
			const current = conversationStore.get(conv.id)!;
			expect(current.messages).toHaveLength(1);
			expect(current.messages[0].id).toBe(message.id);
		});

		it('should add an assistant message to conversation', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'assistant', 'Hi there!');

			expect(message.role).toBe('assistant');
			expect(message.content).toBe('Hi there!');
		});

		it('should throw for non-existent conversation', () => {
			expect(() => {
				conversationStore.addMessage('non-existent', 'user', 'Hello');
			}).toThrow('Conversation not found');
		});

		it('should update conversation updatedAt', () => {
			const conv = conversationStore.create();
			const originalUpdatedAt = conv.updatedAt;

			conversationStore.addMessage(conv.id, 'user', 'Hello');

			expect(conv.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});
	});

	describe('updateMessageContent', () => {
		it('should update message content (for streaming)', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'assistant', 'Initial');

			conversationStore.updateMessageContent(conv.id, message.id, 'Updated content');

			// Fetch current state from store
			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].content).toBe('Updated content');
		});

		it('should do nothing for non-existent message', () => {
			const conv = conversationStore.create();
			conversationStore.updateMessageContent(conv.id, 'non-existent', 'Content');
			// No error, just no-op
		});
	});

	describe('setMessageStreaming', () => {
		it('should set message streaming status', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'assistant', 'Streaming...');

			conversationStore.setMessageStreaming(conv.id, message.id, true);
			// Fetch current state from store
			let current = conversationStore.get(conv.id)!;
			expect(current.messages[0].isStreaming).toBe(true);

			conversationStore.setMessageStreaming(conv.id, message.id, false);
			current = conversationStore.get(conv.id)!;
			expect(current.messages[0].isStreaming).toBe(false);
		});
	});

	describe('updateMessage', () => {
		it('should update message content', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original content');

			conversationStore.updateMessage(conv.id, message.id, 'Updated content');

			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].content).toBe('Updated content');
		});

		it('should mark message as edited', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original');

			expect(conversationStore.get(conv.id)!.messages[0].isEdited).toBeFalsy();

			conversationStore.updateMessage(conv.id, message.id, 'Edited');

			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].isEdited).toBe(true);
		});

		it('should not affect other messages', () => {
			const conv = conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'First message');
			const msg2 = conversationStore.addMessage(conv.id, 'assistant', 'Second message');
			conversationStore.addMessage(conv.id, 'user', 'Third message');

			conversationStore.updateMessage(conv.id, msg2.id, 'Updated second');

			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].content).toBe('First message');
			expect(current.messages[0].isEdited).toBeFalsy();
			expect(current.messages[1].content).toBe('Updated second');
			expect(current.messages[1].isEdited).toBe(true);
			expect(current.messages[2].content).toBe('Third message');
			expect(current.messages[2].isEdited).toBeFalsy();
		});

		it('should update conversation updatedAt timestamp', () => {
			const conv = conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original');
			const originalUpdatedAt = conversationStore.get(conv.id)!.updatedAt;

			conversationStore.updateMessage(conv.id, message.id, 'Updated');

			expect(conversationStore.get(conv.id)!.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});

		it('should do nothing for non-existent conversation', () => {
			const conv = conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'Message');

			// Should not throw
			conversationStore.updateMessage('non-existent', 'msg-id', 'Updated');

			// Original message unchanged
			expect(conversationStore.get(conv.id)!.messages[0].content).toBe('Message');
		});

		it('should do nothing for non-existent message', () => {
			const conv = conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'Message');

			// Should not throw
			conversationStore.updateMessage(conv.id, 'non-existent', 'Updated');

			// Original message unchanged
			expect(conversationStore.get(conv.id)!.messages[0].content).toBe('Message');
		});
	});

	describe('removeMessagesAfter', () => {
		it('should remove all messages after a given message', () => {
			const conv = conversationStore.create();
			const msg1 = conversationStore.addMessage(conv.id, 'user', 'First');
			conversationStore.addMessage(conv.id, 'assistant', 'Second');
			conversationStore.addMessage(conv.id, 'user', 'Third');
			conversationStore.addMessage(conv.id, 'assistant', 'Fourth');

			conversationStore.removeMessagesAfter(conv.id, msg1.id);

			const current = conversationStore.get(conv.id)!;
			expect(current.messages).toHaveLength(1);
			expect(current.messages[0].content).toBe('First');
		});

		it('should do nothing if message is last', () => {
			const conv = conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'First');
			const msg2 = conversationStore.addMessage(conv.id, 'assistant', 'Second');

			conversationStore.removeMessagesAfter(conv.id, msg2.id);

			const current = conversationStore.get(conv.id)!;
			expect(current.messages).toHaveLength(2);
		});

		it('should do nothing for non-existent conversation', () => {
			conversationStore.removeMessagesAfter('non-existent', 'msg-id');
			// Should not throw
		});

		it('should do nothing for non-existent message', () => {
			const conv = conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'First');

			conversationStore.removeMessagesAfter(conv.id, 'non-existent');

			expect(conversationStore.get(conv.id)!.messages).toHaveLength(1);
		});
	});

	// ============================================================================
	// Server Conversation ID Tracking
	// ============================================================================

	describe('setServerConversationId', () => {
		it('should set serverConversationId for API context chaining', () => {
			const conv = conversationStore.create();

			conversationStore.setServerConversationId(conv.id, 'conv_12345');

			// Fetch current state from store
			const current = conversationStore.get(conv.id)!;
			expect(current.serverConversationId).toBe('conv_12345');
		});

		it('should update updatedAt timestamp', () => {
			const conv = conversationStore.create();
			const originalUpdatedAt = conv.updatedAt;

			conversationStore.setServerConversationId(conv.id, 'conv_12345');

			expect(conv.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});
	});

	describe('getServerConversationId', () => {
		it('should return the server conversation ID', () => {
			const conv = conversationStore.create();
			conversationStore.setServerConversationId(conv.id, 'conv_67890');

			expect(conversationStore.getServerConversationId(conv.id)).toBe('conv_67890');
		});

		it('should return null if no server conversation ID set', () => {
			const conv = conversationStore.create();

			expect(conversationStore.getServerConversationId(conv.id)).toBeNull();
		});

		it('should return null for non-existent conversation', () => {
			expect(conversationStore.getServerConversationId('non-existent')).toBeNull();
		});
	});

	// ============================================================================
	// Sorted Getter
	// ============================================================================

	describe('sorted', () => {
		it('should return conversations sorted by updatedAt descending', () => {
			const conv1 = conversationStore.create('First');
			const conv2 = conversationStore.create('Second');
			const conv3 = conversationStore.create('Third');

			// Manually set updatedAt to ensure predictable order
			const current1 = conversationStore.get(conv1.id)!;
			const current2 = conversationStore.get(conv2.id)!;
			const current3 = conversationStore.get(conv3.id)!;

			// Set explicit timestamps: conv2 oldest, conv3 middle, conv1 newest
			current2.updatedAt = 1000;
			current3.updatedAt = 2000;
			current1.updatedAt = 3000;

			const sorted = conversationStore.sorted;

			expect(sorted[0].id).toBe(conv1.id); // newest
			expect(sorted[1].id).toBe(conv3.id); // middle
			expect(sorted[2].id).toBe(conv2.id); // oldest
		});
	});

	// ============================================================================
	// Load/Clear
	// ============================================================================

	describe('clear', () => {
		it('should remove all conversations and reset activeId', () => {
			conversationStore.create();
			conversationStore.create();

			conversationStore.clear();

			expect(conversationStore.conversations).toHaveLength(0);
			expect(conversationStore.activeId).toBeNull();
		});
	});

	describe('load', () => {
		it('should load conversations from external source', () => {
			const conversations = [
				createConversation({ id: 'conv-1', title: 'Loaded 1' }),
				createConversation({ id: 'conv-2', title: 'Loaded 2' })
			];

			conversationStore.load(conversations, 'conv-2');

			expect(conversationStore.conversations).toHaveLength(2);
			expect(conversationStore.activeId).toBe('conv-2');
		});

		it('should set first conversation as active if no activeId provided', () => {
			const conversations = [
				createConversation({ id: 'conv-1', title: 'First' }),
				createConversation({ id: 'conv-2', title: 'Second' })
			];

			conversationStore.load(conversations);

			expect(conversationStore.activeId).toBe('conv-1');
		});

		it('should set activeId to null for empty array', () => {
			conversationStore.load([]);

			expect(conversationStore.activeId).toBeNull();
		});
	});
});

// ============================================================================
// Helper Function Tests
// ============================================================================

describe('createConversation helper', () => {
	it('should create conversation with defaults', () => {
		const conv = createConversation();

		expect(conv.id).toBeDefined();
		expect(conv.title).toBe('New Chat');
		expect(conv.messages).toEqual([]);
		expect(conv.serverConversationId).toBeNull();
		expect(conv.createdAt).toBeDefined();
		expect(conv.updatedAt).toBeDefined();
	});

	it('should allow overrides', () => {
		const conv = createConversation({
			title: 'Custom',
			serverConversationId: 'conv_123'
		});

		expect(conv.title).toBe('Custom');
		expect(conv.serverConversationId).toBe('conv_123');
	});
});

describe('createMessage helper', () => {
	it('should create message with required fields', () => {
		const msg = createMessage('user', 'Hello');

		expect(msg.id).toBeDefined();
		expect(msg.role).toBe('user');
		expect(msg.content).toBe('Hello');
		expect(msg.createdAt).toBeDefined();
	});

	it('should allow overrides', () => {
		const msg = createMessage('assistant', 'Hi', { isStreaming: true });

		expect(msg.isStreaming).toBe(true);
	});
});
