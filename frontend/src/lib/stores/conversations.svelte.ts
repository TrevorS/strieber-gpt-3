/**
 * Conversation State Store
 *
 * Svelte 5 runes-based store for managing chat conversations.
 * Uses $state for reactive state and $derived for computed values.
 */

import {
	type Conversation,
	createConversation,
	createMessage,
	type Message,
	type ResponseOutputItem
} from './types';
import type { Attachment } from '$lib/utils/files';
import { logger } from '$lib/utils/logger';

/**
 * Conversation store class using Svelte 5 runes.
 *
 * @example
 * ```svelte
 * <script>
 *   import { conversationStore } from '$lib/stores';
 *
 *   // Access reactive state directly
 *   const conv = conversationStore.active;
 *   const messages = conv?.messages ?? [];
 * </script>
 * ```
 */
class ConversationStore {
	/** All conversations */
	conversations = $state<Conversation[]>([]);

	/** ID of the currently active conversation */
	activeId = $state<string | null>(null);

	/** The currently active conversation (derived) */
	get active(): Conversation | undefined {
		return this.conversations.find((c) => c.id === this.activeId);
	}

	/** Sorted conversations by updatedAt descending (derived) */
	get sorted(): Conversation[] {
		return [...this.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
	}

	/**
	 * Create a new conversation and set it as active.
	 */
	create(title?: string): Conversation {
		const conv = createConversation(title ? { title } : undefined);
		this.conversations.push(conv);
		this.activeId = conv.id;
		logger.store.action('create', { id: conv.id, title: conv.title, activeId: this.activeId });
		return conv;
	}

	/**
	 * Delete a conversation by ID.
	 * If deleting the active conversation, switches to the most recent one.
	 */
	delete(id: string): void {
		const index = this.conversations.findIndex((c) => c.id === id);
		if (index === -1) {
			logger.warn('store', 'Delete failed: conversation not found', { id });
			return;
		}

		const wasActive = this.activeId === id;
		this.conversations.splice(index, 1);

		// If we deleted the active conversation, switch to another
		if (wasActive) {
			this.activeId = this.sorted[0]?.id ?? null;
		}

		logger.store.action('delete', { id, wasActive, newActiveId: this.activeId });
	}

	/**
	 * Set the active conversation by ID.
	 */
	setActive(id: string | null): void {
		const oldId = this.activeId;
		this.activeId = id;
		logger.store.action('setActive', { oldActiveId: oldId, newActiveId: id });
	}

	/**
	 * Update a conversation's title.
	 */
	updateTitle(id: string, title: string): void {
		const conv = this.conversations.find((c) => c.id === id);
		if (conv) {
			const oldTitle = conv.title;
			conv.title = title;
			conv.updatedAt = Date.now();
			logger.store.action('updateTitle', { id, oldTitle, newTitle: title });
		} else {
			logger.warn('store', 'updateTitle failed: conversation not found', { id });
		}
	}

	/**
	 * Add a message to a conversation.
	 */
	addMessage(
		conversationId: string,
		role: Message['role'],
		content: string,
		attachments?: Attachment[]
	): Message {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.error('store', 'addMessage failed: conversation not found', { conversationId });
			throw new Error(`Conversation not found: ${conversationId}`);
		}

		const message = createMessage(role, content, attachments?.length ? { attachments } : undefined);
		conv.messages.push(message);
		conv.updatedAt = Date.now();
		logger.store.action('addMessage', {
			conversationId,
			messageId: message.id,
			role,
			contentLength: content.length,
			attachmentCount: attachments?.length ?? 0,
			messageCount: conv.messages.length
		});
		return message;
	}

	/**
	 * Update a message's content (used for streaming).
	 */
	updateMessageContent(conversationId: string, messageId: string, content: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.warn('store', 'updateMessageContent: conversation not found', { conversationId });
			return;
		}

		const message = conv.messages.find((m) => m.id === messageId);
		if (message) {
			message.content = content;
			conv.updatedAt = Date.now();
			// Debug level since this is called frequently during streaming
			logger.debug('streaming', 'Message content updated', {
				conversationId,
				messageId,
				contentLength: content.length
			});
		}
	}

	/**
	 * Set a message's streaming status.
	 */
	setMessageStreaming(conversationId: string, messageId: string, isStreaming: boolean): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) return;

		const message = conv.messages.find((m) => m.id === messageId);
		if (message) {
			message.isStreaming = isStreaming;
			logger.info('streaming', isStreaming ? 'Streaming started' : 'Streaming ended', {
				conversationId,
				messageId
			});
		}
	}

	/**
	 * Update the lastResponseId for context chaining.
	 */
	updateLastResponseId(conversationId: string, responseId: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (conv) {
			const oldResponseId = conv.lastResponseId;
			conv.lastResponseId = responseId;
			conv.updatedAt = Date.now();
			logger.info('store', 'Context chain updated', {
				conversationId,
				oldResponseId,
				newResponseId: responseId
			});
		}
	}

	/**
	 * Add or update an output item on a message (used during streaming for tool calls).
	 * If an item with the same ID exists, it will be updated; otherwise, it will be added.
	 */
	setOutputItem(conversationId: string, messageId: string, item: ResponseOutputItem): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.warn('store', 'setOutputItem: conversation not found', { conversationId });
			return;
		}

		const message = conv.messages.find((m) => m.id === messageId);
		if (!message) {
			logger.warn('store', 'setOutputItem: message not found', { conversationId, messageId });
			return;
		}

		// Initialize rawOutput if not present
		if (!message.rawOutput) {
			message.rawOutput = [];
		}

		// Check if item already exists by ID
		const itemId = 'id' in item ? item.id : undefined;
		if (itemId) {
			const existingIndex = message.rawOutput.findIndex(
				(existing) => 'id' in existing && existing.id === itemId
			);

			if (existingIndex !== -1) {
				// Update existing item
				message.rawOutput[existingIndex] = item;
				logger.debug('store', 'Output item updated', {
					conversationId,
					messageId,
					itemId,
					itemType: item.type
				});
				return;
			}
		}

		// Add new item
		message.rawOutput.push(item);
		logger.debug('store', 'Output item added', {
			conversationId,
			messageId,
			itemId,
			itemType: item.type,
			totalItems: message.rawOutput.length
		});
	}

	/**
	 * Clear all output items from a message.
	 */
	clearOutputItems(conversationId: string, messageId: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) return;

		const message = conv.messages.find((m) => m.id === messageId);
		if (message) {
			message.rawOutput = [];
			logger.debug('store', 'Output items cleared', { conversationId, messageId });
		}
	}

	/**
	 * Remove the last assistant message and return the preceding user message text.
	 * Used for regenerating a response.
	 */
	removeLastAssistantMessage(conversationId: string): string | null {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv || conv.messages.length < 2) {
			logger.warn('store', 'removeLastAssistantMessage: not enough messages', { conversationId });
			return null;
		}

		const lastMessage = conv.messages[conv.messages.length - 1];
		if (lastMessage.role !== 'assistant') {
			logger.warn('store', 'removeLastAssistantMessage: last message is not assistant', {
				conversationId,
				lastRole: lastMessage.role
			});
			return null;
		}

		// Get the user message that preceded the assistant message
		const userMessage = conv.messages[conv.messages.length - 2];
		if (userMessage.role !== 'user') {
			logger.warn('store', 'removeLastAssistantMessage: preceding message is not user', {
				conversationId,
				precedingRole: userMessage.role
			});
			return null;
		}

		// Remove the assistant message
		conv.messages.pop();
		conv.updatedAt = Date.now();

		logger.store.action('removeLastAssistantMessage', {
			conversationId,
			removedMessageId: lastMessage.id,
			userMessageText: userMessage.content.slice(0, 50)
		});

		return userMessage.content;
	}

	/**
	 * Get a conversation by ID.
	 */
	get(id: string): Conversation | undefined {
		return this.conversations.find((c) => c.id === id);
	}

	/**
	 * Clear all conversations.
	 */
	clear(): void {
		const count = this.conversations.length;
		this.conversations = [];
		this.activeId = null;
		logger.store.action('clear', { conversationsCleared: count });
	}

	/**
	 * Load conversations from external source (e.g., localStorage).
	 * If activeId is not provided, defaults to the first conversation.
	 * Pass null for activeId to explicitly have no active conversation.
	 */
	load(conversations: Conversation[], activeId?: string | null): void {
		this.conversations = conversations;
		// Default to first conversation if activeId not explicitly provided
		this.activeId = activeId !== undefined ? activeId : (conversations[0]?.id ?? null);
		logger.info('persistence', 'Conversations loaded from storage', {
			conversationCount: conversations.length,
			activeId: this.activeId,
			conversationIds: conversations.map((c) => c.id)
		});
	}
}

/**
 * Singleton conversation store instance.
 */
export const conversationStore = new ConversationStore();
