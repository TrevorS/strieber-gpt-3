/**
 * Conversation State Store
 *
 * Svelte 5 runes-based store for managing chat conversations.
 * Uses $state for reactive state and $derived for computed values.
 * Syncs with server Conversations API - server is source of truth.
 */

import {
	type Conversation,
	serverToLocalConversation,
	createMessage,
	type Message,
	type ResponseOutputItem
} from './types';
import type { Attachment } from '$lib/utils/files';
import { logger } from '$lib/utils/logger';
import {
	listConversations,
	createConversation as apiCreateConversation,
	deleteConversation as apiDeleteConversation,
	updateConversation as apiUpdateConversation,
	listItems,
	type ConversationItem
} from '$lib/api/conversations';

/**
 * Conversation store class using Svelte 5 runes.
 * Now backed by server Conversations API.
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
	/** All conversations (cached from server) */
	conversations = $state<Conversation[]>([]);

	/** ID of the currently active conversation */
	activeId = $state<string | null>(null);

	/** Loading state for initial fetch */
	isLoading = $state(false);

	/** Error from last operation */
	error = $state<string | null>(null);

	/** The currently active conversation (derived) */
	get active(): Conversation | undefined {
		return this.conversations.find((c) => c.id === this.activeId);
	}

	/** Sorted conversations by updatedAt descending (derived) */
	get sorted(): Conversation[] {
		return [...this.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
	}

	/**
	 * Fetch all conversations from the server.
	 * Called on app startup to populate the store.
	 * Note: This only loads metadata (no items/messages) for sidebar display.
	 * Use loadItems() to fetch messages when navigating to a conversation.
	 */
	async fetchAll(): Promise<void> {
		this.isLoading = true;
		this.error = null;

		try {
			const response = await listConversations({ limit: 100, order: 'desc' });
			this.conversations = response.data.map((server) => serverToLocalConversation(server));
			logger.info('persistence', 'Conversations loaded from server', {
				conversationCount: this.conversations.length,
				conversationIds: this.conversations.map((c) => c.id)
			});
		} catch (e) {
			const message = e instanceof Error ? e.message : 'Failed to fetch conversations';
			this.error = message;
			logger.error('persistence', 'Failed to load conversations from server', { error: message });
		} finally {
			this.isLoading = false;
		}
	}

	/**
	 * Load items (messages) for a specific conversation from the server.
	 * Calls the /conversations/{id}/items endpoint and converts to local messages.
	 */
	async loadItems(conversationId: string): Promise<void> {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.warn('store', 'loadItems: conversation not found locally', { conversationId });
			return;
		}

		// Skip if we already have messages (they were added locally during this session)
		if (conv.messages.length > 0) {
			logger.debug('store', 'loadItems: conversation already has messages', {
				conversationId,
				messageCount: conv.messages.length
			});
			return;
		}

		try {
			// Use the /items endpoint directly instead of include param
			const itemList = await listItems(conversationId, { limit: 100, order: 'asc' });
			if (itemList.data && itemList.data.length > 0) {
				const messages = this.itemsToMessages(itemList.data);
				// Replace the conversation object in the array to trigger Svelte 5 reactivity
				// (mutating conv.messages in place doesn't trigger $derived updates)
				const index = this.conversations.findIndex((c) => c.id === conversationId);
				if (index !== -1) {
					this.conversations[index] = { ...conv, messages };
					// Force Svelte 5 reactivity for $derived values across modules
					this.conversations = [...this.conversations];
				}
				logger.info('persistence', 'Items loaded for conversation', {
					conversationId,
					itemCount: itemList.data.length,
					messageCount: messages.length
				});
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : 'Failed to load items';
			logger.error('persistence', 'Failed to load items for conversation', {
				conversationId,
				error: message
			});
			// If 404, the conversation may have been deleted on server
			if (message.includes('not found')) {
				this.removeLocal(conversationId);
			}
			throw e;
		}
	}

	/**
	 * Convert server conversation items to local Message objects.
	 * Groups related items (message + output items) into single messages.
	 */
	private itemsToMessages(items: ConversationItem[]): Message[] {
		const messages: Message[] = [];
		let currentAssistantMessage: Message | null = null;

		for (const item of items) {
			const itemType = item.type;

			// User message
			if (itemType === 'message' && item.role === 'user') {
				// If we have a pending assistant message, push it
				if (currentAssistantMessage) {
					messages.push(currentAssistantMessage);
					currentAssistantMessage = null;
				}

				// Extract text content from the message
				const content = this.extractMessageText(item);
				messages.push(createMessage('user', content, { id: item.id }));
			}
			// Assistant message
			else if (itemType === 'message' && item.role === 'assistant') {
				// If we have a pending assistant message, push it first
				if (currentAssistantMessage) {
					messages.push(currentAssistantMessage);
				}

				const content = this.extractMessageText(item);
				currentAssistantMessage = createMessage('assistant', content, {
					id: item.id,
					rawOutput: []
				});
			}
			// Tool calls, reasoning, etc. - attach to current assistant message
			else if (
				itemType === 'function_call' ||
				itemType === 'web_search_call' ||
				itemType === 'code_interpreter_call' ||
				itemType === 'reasoning' ||
				itemType === 'file_search_call' ||
				itemType === 'computer_call'
			) {
				// Start a new assistant message if we don't have one
				if (!currentAssistantMessage) {
					currentAssistantMessage = createMessage('assistant', '', { rawOutput: [] });
				}
				// Add to rawOutput - cast item to ResponseOutputItem (close enough for display)
				currentAssistantMessage.rawOutput?.push(item as unknown as ResponseOutputItem);
			}
			// Function call output - attach to current assistant message
			else if (itemType === 'function_call_output') {
				if (currentAssistantMessage?.rawOutput) {
					// Find the matching function_call and attach output
					const funcCall = currentAssistantMessage.rawOutput.find(
						(o) => o.type === 'function_call' && 'call_id' in o && o.call_id === item.call_id
					);
					if (funcCall && 'output' in funcCall) {
						// The output is already in the function_call item from server
					}
				}
			}
		}

		// Push any remaining assistant message
		if (currentAssistantMessage) {
			messages.push(currentAssistantMessage);
		}

		return messages;
	}

	/**
	 * Extract text content from a message item.
	 */
	private extractMessageText(item: ConversationItem): string {
		// Content can be an array of content parts or a string
		const content = item.content;
		if (typeof content === 'string') {
			return content;
		}
		if (Array.isArray(content)) {
			// Extract text from content parts
			return content
				.filter(
					(part: { type?: string }) =>
						part.type === 'input_text' || part.type === 'output_text' || part.type === 'text'
				)
				.map((part: { text?: string }) => part.text || '')
				.join('');
		}
		return '';
	}

	/**
	 * Create a new conversation on the server and add to local state.
	 * @param title - Optional title (stored in metadata)
	 * @returns The created conversation
	 */
	async create(title?: string): Promise<Conversation> {
		const metadata = title ? { title } : undefined;
		const serverConv = await apiCreateConversation(metadata);
		const conv = serverToLocalConversation(serverConv);

		this.conversations.push(conv);
		this.activeId = conv.id;

		logger.store.action('create', { id: conv.id, title: conv.title, activeId: this.activeId });
		return conv;
	}

	/**
	 * Delete a conversation from server and local state.
	 * If deleting the active conversation, switches to the most recent one.
	 */
	async delete(id: string): Promise<void> {
		const index = this.conversations.findIndex((c) => c.id === id);
		if (index === -1) {
			logger.warn('store', 'Delete failed: conversation not found locally', { id });
			return;
		}

		try {
			await apiDeleteConversation(id);
		} catch (e) {
			// If 404, it's already deleted on server - that's fine
			const message = e instanceof Error ? e.message : 'Unknown error';
			if (!message.includes('not found')) {
				throw e;
			}
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
	 * Remove a conversation from local state only (e.g., when server returns 404).
	 * Does not call API.
	 */
	removeLocal(id: string): void {
		const index = this.conversations.findIndex((c) => c.id === id);
		if (index === -1) return;

		const wasActive = this.activeId === id;
		this.conversations.splice(index, 1);

		if (wasActive) {
			this.activeId = this.sorted[0]?.id ?? null;
		}

		logger.store.action('removeLocal', { id, wasActive, newActiveId: this.activeId });
	}

	/**
	 * Set the active conversation by ID.
	 */
	setActive(id: string | null): void {
		const oldId = this.activeId;
		this.activeId = id;
		// Force Svelte 5 reactivity for $derived values across modules
		this.conversations = [...this.conversations];
		logger.store.action('setActive', { oldActiveId: oldId, newActiveId: id });
	}

	/**
	 * Update a conversation's title (syncs to server metadata).
	 */
	async updateTitle(id: string, title: string): Promise<void> {
		const conv = this.conversations.find((c) => c.id === id);
		if (!conv) {
			logger.warn('store', 'updateTitle failed: conversation not found', { id });
			return;
		}

		const oldTitle = conv.title;

		try {
			await apiUpdateConversation(id, { title });
			conv.title = title;
			conv.updatedAt = Date.now();
			logger.store.action('updateTitle', { id, oldTitle, newTitle: title });
		} catch (e) {
			logger.error('store', 'updateTitle failed', {
				id,
				error: e instanceof Error ? e.message : 'Unknown error'
			});
			throw e;
		}
	}

	/**
	 * Update title locally only (for optimistic updates during streaming).
	 * Use updateTitle() for persisted updates.
	 */
	updateTitleLocal(id: string, title: string): void {
		const conv = this.conversations.find((c) => c.id === id);
		if (conv) {
			conv.title = title;
			conv.updatedAt = Date.now();
		}
	}

	/**
	 * Add a message to a conversation (local state only).
	 * Server sync happens via streaming response.
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
	 * Uses array replacement to trigger Svelte 5 reactivity for $derived values.
	 */
	updateMessageContent(conversationId: string, messageId: string, content: string): void {
		const convIndex = this.conversations.findIndex((c) => c.id === conversationId);
		if (convIndex === -1) {
			logger.warn('store', 'updateMessageContent: conversation not found', { conversationId });
			return;
		}

		const conv = this.conversations[convIndex];
		const msgIndex = conv.messages.findIndex((m) => m.id === messageId);
		if (msgIndex === -1) {
			logger.warn('store', 'updateMessageContent: message not found', {
				conversationId,
				messageId
			});
			return;
		}

		// Create new message with updated content to trigger Svelte 5 reactivity
		const updatedMessages = [...conv.messages];
		updatedMessages[msgIndex] = { ...conv.messages[msgIndex], content };

		// Replace the conversation object to trigger $derived re-computation
		this.conversations[convIndex] = { ...conv, messages: updatedMessages, updatedAt: Date.now() };

		// Debug level since this is called frequently during streaming
		logger.debug('streaming', 'Message content updated', {
			conversationId,
			messageId,
			contentLength: content.length
		});
	}

	/**
	 * Update a message's content and mark it as edited (used for user message editing).
	 */
	updateMessage(conversationId: string, messageId: string, content: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.warn('store', 'updateMessage: conversation not found', { conversationId });
			return;
		}

		const message = conv.messages.find((m) => m.id === messageId);
		if (!message) {
			logger.warn('store', 'updateMessage: message not found', { conversationId, messageId });
			return;
		}

		const oldContent = message.content;
		message.content = content;
		message.isEdited = true;
		conv.updatedAt = Date.now();

		logger.store.action('updateMessage', {
			conversationId,
			messageId,
			oldContentLength: oldContent.length,
			newContentLength: content.length
		});
	}

	/**
	 * Set a message's streaming status.
	 * Uses array replacement to trigger Svelte 5 reactivity for $derived values.
	 */
	setMessageStreaming(conversationId: string, messageId: string, isStreaming: boolean): void {
		const convIndex = this.conversations.findIndex((c) => c.id === conversationId);
		if (convIndex === -1) return;

		const conv = this.conversations[convIndex];
		const msgIndex = conv.messages.findIndex((m) => m.id === messageId);
		if (msgIndex === -1) return;

		// Create new message with updated streaming status to trigger Svelte 5 reactivity
		const updatedMessages = [...conv.messages];
		updatedMessages[msgIndex] = { ...conv.messages[msgIndex], isStreaming };

		// Replace the conversation object to trigger $derived re-computation
		this.conversations[convIndex] = { ...conv, messages: updatedMessages };

		logger.info('streaming', isStreaming ? 'Streaming started' : 'Streaming ended', {
			conversationId,
			messageId
		});
	}

	/**
	 * Add or update an output item on a message (used during streaming for tool calls).
	 * If an item with the same ID exists, it will be updated; otherwise, it will be added.
	 * Uses array replacement to trigger Svelte 5 reactivity for $derived values.
	 */
	setOutputItem(conversationId: string, messageId: string, item: ResponseOutputItem): void {
		const convIndex = this.conversations.findIndex((c) => c.id === conversationId);
		if (convIndex === -1) {
			logger.warn('store', 'setOutputItem: conversation not found', { conversationId });
			return;
		}

		const conv = this.conversations[convIndex];
		const msgIndex = conv.messages.findIndex((m) => m.id === messageId);
		if (msgIndex === -1) {
			logger.warn('store', 'setOutputItem: message not found', { conversationId, messageId });
			return;
		}

		const message = conv.messages[msgIndex];
		const currentRawOutput = message.rawOutput ?? [];
		const itemId = 'id' in item ? item.id : undefined;

		let updatedRawOutput: ResponseOutputItem[];

		// Check if item already exists by ID
		if (itemId) {
			const existingIndex = currentRawOutput.findIndex(
				(existing) => 'id' in existing && existing.id === itemId
			);

			if (existingIndex !== -1) {
				// Update existing item
				updatedRawOutput = [...currentRawOutput];
				updatedRawOutput[existingIndex] = item;
				logger.debug('store', 'Output item updated', {
					conversationId,
					messageId,
					itemId,
					itemType: item.type
				});
			} else {
				// Add new item
				updatedRawOutput = [...currentRawOutput, item];
				logger.debug('store', 'Output item added', {
					conversationId,
					messageId,
					itemId,
					itemType: item.type,
					totalItems: updatedRawOutput.length
				});
			}
		} else {
			// Add new item (no ID to match)
			updatedRawOutput = [...currentRawOutput, item];
			logger.debug('store', 'Output item added', {
				conversationId,
				messageId,
				itemId,
				itemType: item.type,
				totalItems: updatedRawOutput.length
			});
		}

		// Create new message with updated rawOutput to trigger Svelte 5 reactivity
		const updatedMessages = [...conv.messages];
		updatedMessages[msgIndex] = { ...message, rawOutput: updatedRawOutput };

		// Replace the conversation object to trigger $derived re-computation
		this.conversations[convIndex] = { ...conv, messages: updatedMessages };
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
	 * Update function call arguments during streaming.
	 * Finds the function_call item by item_id (the item's id field) and appends to its arguments.
	 * Uses array replacement to trigger Svelte 5 reactivity for $derived values.
	 */
	updateFunctionCallArguments(
		conversationId: string,
		messageId: string,
		itemId: string,
		argumentsDelta: string
	): void {
		const convIndex = this.conversations.findIndex((c) => c.id === conversationId);
		if (convIndex === -1) return;

		const conv = this.conversations[convIndex];
		const msgIndex = conv.messages.findIndex((m) => m.id === messageId);
		if (msgIndex === -1) return;

		const message = conv.messages[msgIndex];
		if (!message.rawOutput) return;

		// Find the function_call item by its id (item_id in delta events)
		const itemIndex = message.rawOutput.findIndex(
			(i) => i.type === 'function_call' && 'id' in i && i.id === itemId
		);

		if (itemIndex === -1) return;

		const item = message.rawOutput[itemIndex];
		if (!('arguments' in item)) return;

		// Create updated item with appended arguments
		const updatedItem = {
			...item,
			arguments: (item as { arguments: string }).arguments + argumentsDelta
		};

		// Create new rawOutput array with updated item
		const updatedRawOutput = [...message.rawOutput];
		updatedRawOutput[itemIndex] = updatedItem as ResponseOutputItem;

		// Create new message with updated rawOutput to trigger Svelte 5 reactivity
		const updatedMessages = [...conv.messages];
		updatedMessages[msgIndex] = { ...message, rawOutput: updatedRawOutput };

		// Replace the conversation object to trigger $derived re-computation
		this.conversations[convIndex] = { ...conv, messages: updatedMessages };

		logger.debug('store', 'Function call arguments updated', {
			conversationId,
			messageId,
			itemId,
			deltaLength: argumentsDelta.length
		});
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
	 * Remove all messages after a given message ID (exclusive).
	 * Used when editing a user message to remove subsequent responses.
	 */
	removeMessagesAfter(conversationId: string, messageId: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			logger.warn('store', 'removeMessagesAfter: conversation not found', { conversationId });
			return;
		}

		const messageIndex = conv.messages.findIndex((m) => m.id === messageId);
		if (messageIndex === -1) {
			logger.warn('store', 'removeMessagesAfter: message not found', { conversationId, messageId });
			return;
		}

		const removedCount = conv.messages.length - messageIndex - 1;
		if (removedCount > 0) {
			conv.messages.splice(messageIndex + 1);
			conv.updatedAt = Date.now();
			logger.store.action('removeMessagesAfter', {
				conversationId,
				messageId,
				removedCount
			});
		}
	}

	/**
	 * Get a conversation by ID.
	 */
	get(id: string): Conversation | undefined {
		return this.conversations.find((c) => c.id === id);
	}

	/**
	 * Clear all conversations from local state.
	 */
	clear(): void {
		const count = this.conversations.length;
		this.conversations = [];
		this.activeId = null;
		logger.store.action('clear', { conversationsCleared: count });
	}

	/**
	 * Add a conversation to local state (used after creating via API).
	 */
	addLocal(conv: Conversation): void {
		// Check if already exists
		if (this.conversations.some((c) => c.id === conv.id)) {
			return;
		}
		this.conversations.push(conv);
		logger.store.action('addLocal', { id: conv.id, title: conv.title });
	}
}

/**
 * Singleton conversation store instance.
 */
export const conversationStore = new ConversationStore();
