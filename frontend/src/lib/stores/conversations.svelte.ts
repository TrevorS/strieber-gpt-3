/**
 * Conversation State Store
 *
 * Svelte 5 runes-based store for managing chat conversations.
 * Uses $state for reactive state and $derived for computed values.
 */

import { type Conversation, createConversation, createMessage, type Message } from './types';

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
		return conv;
	}

	/**
	 * Delete a conversation by ID.
	 * If deleting the active conversation, switches to the most recent one.
	 */
	delete(id: string): void {
		const index = this.conversations.findIndex((c) => c.id === id);
		if (index === -1) return;

		this.conversations.splice(index, 1);

		// If we deleted the active conversation, switch to another
		if (this.activeId === id) {
			this.activeId = this.sorted[0]?.id ?? null;
		}
	}

	/**
	 * Set the active conversation by ID.
	 */
	setActive(id: string | null): void {
		this.activeId = id;
	}

	/**
	 * Update a conversation's title.
	 */
	updateTitle(id: string, title: string): void {
		const conv = this.conversations.find((c) => c.id === id);
		if (conv) {
			conv.title = title;
			conv.updatedAt = Date.now();
		}
	}

	/**
	 * Add a message to a conversation.
	 */
	addMessage(conversationId: string, role: Message['role'], content: string): Message {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) {
			throw new Error(`Conversation not found: ${conversationId}`);
		}

		const message = createMessage(role, content);
		conv.messages.push(message);
		conv.updatedAt = Date.now();
		return message;
	}

	/**
	 * Update a message's content (used for streaming).
	 */
	updateMessageContent(conversationId: string, messageId: string, content: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (!conv) return;

		const message = conv.messages.find((m) => m.id === messageId);
		if (message) {
			message.content = content;
			conv.updatedAt = Date.now();
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
		}
	}

	/**
	 * Update the lastResponseId for context chaining.
	 */
	updateLastResponseId(conversationId: string, responseId: string): void {
		const conv = this.conversations.find((c) => c.id === conversationId);
		if (conv) {
			conv.lastResponseId = responseId;
			conv.updatedAt = Date.now();
		}
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
		this.conversations = [];
		this.activeId = null;
	}

	/**
	 * Load conversations from external source (e.g., localStorage).
	 */
	load(conversations: Conversation[], activeId: string | null = null): void {
		this.conversations = conversations;
		this.activeId = activeId ?? conversations[0]?.id ?? null;
	}
}

/**
 * Singleton conversation store instance.
 */
export const conversationStore = new ConversationStore();
