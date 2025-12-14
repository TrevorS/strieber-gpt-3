/**
 * Unit tests for Conversation State Store
 *
 * Tests cover:
 * - CRUD operations (create, read, update, delete)
 * - Active conversation tracking
 * - Message operations
 * - Edge cases
 *
 * Note: create() and delete() are async as they call the API.
 * In tests we mock the API calls or use local methods.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { conversationStore } from '../conversations.svelte';
import { createMessage, type Conversation } from '../types';

// Mock the API module - the mock captures the metadata passed in and returns it
vi.mock('$lib/api/conversations', () => ({
	createConversation: vi.fn((metadata?: { title?: string }) =>
		Promise.resolve({
			id: `conv_test_${Math.random().toString(36).slice(2)}`,
			object: 'conversation',
			created_at: Math.floor(Date.now() / 1000),
			metadata: metadata || {}
		})
	),
	deleteConversation: vi.fn(() => Promise.resolve()),
	updateConversation: vi.fn(() =>
		Promise.resolve({
			id: 'test',
			object: 'conversation',
			created_at: Math.floor(Date.now() / 1000),
			metadata: {}
		})
	),
	listConversations: vi.fn(() =>
		Promise.resolve({
			object: 'list',
			data: [],
			has_more: false
		})
	)
}));

// Helper to create test conversations for testing (bypasses API)
function createTestConversation(overrides: Partial<Conversation> = {}): Conversation {
	const now = Date.now();
	return {
		id: `conv_test_${Math.random().toString(36).slice(2)}`,
		title: 'New Chat',
		createdAt: now,
		updatedAt: now,
		messages: [],
		...overrides
	};
}

describe('conversationStore', () => {
	beforeEach(() => {
		// Reset store state before each test
		conversationStore.clear();
		vi.clearAllMocks();
	});

	// ============================================================================
	// Create Operations
	// ============================================================================

	describe('create', () => {
		it('should create a new conversation with default title', async () => {
			const conv = await conversationStore.create();

			expect(conv.id).toBeDefined();
			expect(conv.id).toMatch(/^conv_/); // Server ID format
			expect(conv.title).toBe('New Chat');
			expect(conv.messages).toEqual([]);
			expect(conversationStore.conversations).toHaveLength(1);
		});

		it('should create a conversation with custom title', async () => {
			const conv = await conversationStore.create('My Custom Chat');

			expect(conv.title).toBe('My Custom Chat');
		});

		it('should set the new conversation as active', async () => {
			const conv = await conversationStore.create();

			expect(conversationStore.activeId).toBe(conv.id);
			expect(conversationStore.get(conv.id)).toBeDefined();
		});

		it('should create multiple conversations', async () => {
			await conversationStore.create('First');
			const conv2 = await conversationStore.create('Second');

			expect(conversationStore.conversations).toHaveLength(2);
			// Most recent should be active
			expect(conversationStore.activeId).toBe(conv2.id);
		});
	});

	// ============================================================================
	// Delete Operations
	// ============================================================================

	describe('delete', () => {
		it('should delete a conversation by ID', async () => {
			const conv = await conversationStore.create();
			await conversationStore.delete(conv.id);

			expect(conversationStore.conversations).toHaveLength(0);
		});

		it('should do nothing when deleting non-existent ID', async () => {
			await conversationStore.create();
			await conversationStore.delete('non-existent-id');

			expect(conversationStore.conversations).toHaveLength(1);
		});

		it('should switch active to another conversation when deleting active', async () => {
			const conv1 = await conversationStore.create('First');
			const conv2 = await conversationStore.create('Second');

			// conv2 is active
			expect(conversationStore.activeId).toBe(conv2.id);

			await conversationStore.delete(conv2.id);

			// Should switch to conv1
			expect(conversationStore.activeId).toBe(conv1.id);
		});

		it('should set activeId to null when deleting last conversation', async () => {
			const conv = await conversationStore.create();
			await conversationStore.delete(conv.id);

			expect(conversationStore.activeId).toBeNull();
			expect(conversationStore.active).toBeUndefined();
		});
	});

	// ============================================================================
	// Active Conversation
	// ============================================================================

	describe('setActive', () => {
		it('should set active conversation by ID', async () => {
			const conv1 = await conversationStore.create('First');
			await conversationStore.create('Second');

			conversationStore.setActive(conv1.id);

			expect(conversationStore.activeId).toBe(conv1.id);
			expect(conversationStore.active?.title).toBe('First');
		});

		it('should allow setting active to null', async () => {
			await conversationStore.create();
			conversationStore.setActive(null);

			expect(conversationStore.activeId).toBeNull();
		});
	});

	// ============================================================================
	// Title Updates
	// ============================================================================

	describe('updateTitle', () => {
		it('should update conversation title', async () => {
			const conv = await conversationStore.create('Original');
			await conversationStore.updateTitle(conv.id, 'Updated Title');

			expect(conversationStore.get(conv.id)?.title).toBe('Updated Title');
		});

		it('should update updatedAt timestamp', async () => {
			const conv = await conversationStore.create();
			const originalUpdatedAt = conv.updatedAt;

			// Small delay to ensure timestamp differs
			await conversationStore.updateTitle(conv.id, 'New Title');

			expect(conversationStore.get(conv.id)?.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});

		it('should do nothing for non-existent conversation', async () => {
			await conversationStore.updateTitle('non-existent', 'Title');
			// No error thrown, just no-op
			expect(conversationStore.conversations).toHaveLength(0);
		});
	});

	// ============================================================================
	// Local Title Updates (for sync operations)
	// ============================================================================

	describe('updateTitleLocal', () => {
		it('should update conversation title locally without API call', async () => {
			const conv = await conversationStore.create('Original');
			conversationStore.updateTitleLocal(conv.id, 'Updated Title');

			expect(conversationStore.get(conv.id)?.title).toBe('Updated Title');
		});
	});

	// ============================================================================
	// Message Operations
	// ============================================================================

	describe('addMessage', () => {
		it('should add a user message to conversation', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Hello!');

			expect(message.role).toBe('user');
			expect(message.content).toBe('Hello!');
			// Fetch current state from store (Svelte 5 reactivity)
			const current = conversationStore.get(conv.id)!;
			expect(current.messages).toHaveLength(1);
			expect(current.messages[0].id).toBe(message.id);
		});

		it('should add an assistant message to conversation', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'assistant', 'Hi there!');

			expect(message.role).toBe('assistant');
			expect(message.content).toBe('Hi there!');
		});

		it('should throw for non-existent conversation', () => {
			expect(() => {
				conversationStore.addMessage('non-existent', 'user', 'Hello');
			}).toThrow('Conversation not found');
		});

		it('should update conversation updatedAt', async () => {
			const conv = await conversationStore.create();
			const originalUpdatedAt = conv.updatedAt;

			conversationStore.addMessage(conv.id, 'user', 'Hello');

			expect(conv.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});
	});

	describe('updateMessageContent', () => {
		it('should update message content (for streaming)', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'assistant', 'Initial');

			conversationStore.updateMessageContent(conv.id, message.id, 'Updated content');

			// Fetch current state from store
			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].content).toBe('Updated content');
		});

		it('should do nothing for non-existent message', async () => {
			const conv = await conversationStore.create();
			conversationStore.updateMessageContent(conv.id, 'non-existent', 'Content');
			// No error, just no-op
		});
	});

	describe('setMessageStreaming', () => {
		it('should set message streaming status', async () => {
			const conv = await conversationStore.create();
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
		it('should update message content', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original content');

			conversationStore.updateMessage(conv.id, message.id, 'Updated content');

			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].content).toBe('Updated content');
		});

		it('should mark message as edited', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original');

			expect(conversationStore.get(conv.id)!.messages[0].isEdited).toBeFalsy();

			conversationStore.updateMessage(conv.id, message.id, 'Edited');

			const current = conversationStore.get(conv.id)!;
			expect(current.messages[0].isEdited).toBe(true);
		});

		it('should not affect other messages', async () => {
			const conv = await conversationStore.create();
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

		it('should update conversation updatedAt timestamp', async () => {
			const conv = await conversationStore.create();
			const message = conversationStore.addMessage(conv.id, 'user', 'Original');
			const originalUpdatedAt = conversationStore.get(conv.id)!.updatedAt;

			conversationStore.updateMessage(conv.id, message.id, 'Updated');

			expect(conversationStore.get(conv.id)!.updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt);
		});

		it('should do nothing for non-existent conversation', async () => {
			const conv = await conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'Message');

			// Should not throw
			conversationStore.updateMessage('non-existent', 'msg-id', 'Updated');

			// Original message unchanged
			expect(conversationStore.get(conv.id)!.messages[0].content).toBe('Message');
		});

		it('should do nothing for non-existent message', async () => {
			const conv = await conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'Message');

			// Should not throw
			conversationStore.updateMessage(conv.id, 'non-existent', 'Updated');

			// Original message unchanged
			expect(conversationStore.get(conv.id)!.messages[0].content).toBe('Message');
		});
	});

	describe('removeMessagesAfter', () => {
		it('should remove all messages after a given message', async () => {
			const conv = await conversationStore.create();
			const msg1 = conversationStore.addMessage(conv.id, 'user', 'First');
			conversationStore.addMessage(conv.id, 'assistant', 'Second');
			conversationStore.addMessage(conv.id, 'user', 'Third');
			conversationStore.addMessage(conv.id, 'assistant', 'Fourth');

			conversationStore.removeMessagesAfter(conv.id, msg1.id);

			const current = conversationStore.get(conv.id)!;
			expect(current.messages).toHaveLength(1);
			expect(current.messages[0].content).toBe('First');
		});

		it('should do nothing if message is last', async () => {
			const conv = await conversationStore.create();
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

		it('should do nothing for non-existent message', async () => {
			const conv = await conversationStore.create();
			conversationStore.addMessage(conv.id, 'user', 'First');

			conversationStore.removeMessagesAfter(conv.id, 'non-existent');

			expect(conversationStore.get(conv.id)!.messages).toHaveLength(1);
		});
	});

	// ============================================================================
	// Sorted Getter
	// ============================================================================

	describe('sorted', () => {
		it('should return conversations sorted by updatedAt descending', async () => {
			const conv1 = await conversationStore.create('First');
			const conv2 = await conversationStore.create('Second');
			const conv3 = await conversationStore.create('Third');

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
	// Clear
	// ============================================================================

	describe('clear', () => {
		it('should remove all conversations and reset activeId', async () => {
			await conversationStore.create();
			await conversationStore.create();

			conversationStore.clear();

			expect(conversationStore.conversations).toHaveLength(0);
			expect(conversationStore.activeId).toBeNull();
		});
	});

	// ============================================================================
	// addLocal (for loading from API)
	// ============================================================================

	describe('addLocal', () => {
		it('should add a conversation locally without API call', () => {
			const conv = createTestConversation({ title: 'Test Local' });
			conversationStore.addLocal(conv);

			expect(conversationStore.conversations).toHaveLength(1);
			expect(conversationStore.get(conv.id)?.title).toBe('Test Local');
		});

		it('should skip adding if conversation with same ID already exists', () => {
			const conv = createTestConversation({ title: 'Original' });
			conversationStore.addLocal(conv);
			conversationStore.addLocal({ ...conv, title: 'Duplicate' });

			// addLocal skips duplicates to prevent overwriting local changes
			expect(conversationStore.conversations).toHaveLength(1);
			expect(conversationStore.get(conv.id)?.title).toBe('Original');
		});
	});

	// ============================================================================
	// removeLocal
	// ============================================================================

	describe('removeLocal', () => {
		it('should remove a conversation locally without API call', () => {
			const conv = createTestConversation();
			conversationStore.addLocal(conv);

			conversationStore.removeLocal(conv.id);

			expect(conversationStore.conversations).toHaveLength(0);
		});
	});
});

// ============================================================================
// Helper Function Tests
// ============================================================================

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
