/**
 * Conversations API Client
 *
 * Functions for interacting with the Conversations API endpoints.
 * Conversations provide server-side state management for multi-turn interactions.
 */

import { getApiBaseUrl } from './client';
import { logger } from '$lib/utils/logger';

// ============================================================================
// Types
// ============================================================================

/**
 * Server-side conversation object
 */
export interface ServerConversation {
	/** Unique identifier (conv_xxx) */
	id: string;
	/** Object type, always "conversation" */
	object: 'conversation';
	/** Unix timestamp of creation */
	created_at: number;
	/** Optional metadata */
	metadata?: Record<string, string>;
	/** Items (only present when include=conversation.items) */
	items?: ConversationItem[];
}

/**
 * A stored item in a conversation
 */
export interface ConversationItem {
	/** Unique item ID */
	id: string;
	/** Item status */
	status: 'completed' | 'in_progress' | 'incomplete';
	/** Item type and data (flattened) */
	type: string;
	[key: string]: unknown;
}

/**
 * Paginated list of conversation items
 */
export interface ItemList {
	object: 'list';
	data: ConversationItem[];
	first_id?: string;
	last_id?: string;
	has_more: boolean;
}

/**
 * Deletion response
 */
export interface ConversationDeleted {
	id: string;
	object: 'conversation.deleted';
	deleted: boolean;
}

/**
 * Pagination options for listing items
 */
export interface PaginationOptions {
	after?: string;
	limit?: number;
	order?: 'asc' | 'desc';
	include?: string[];
}

/**
 * Input item for creating items in a conversation
 */
export interface InputItem {
	type: string;
	[key: string]: unknown;
}

/**
 * List response for conversations
 */
export interface ConversationListResponse {
	object: 'list';
	data: ServerConversation[];
	first_id?: string;
	last_id?: string;
	has_more: boolean;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * List all conversations on the server.
 *
 * @param options - Pagination options
 * @returns Paginated list of conversations
 */
export async function listConversations(
	options: { limit?: number; order?: 'asc' | 'desc'; after?: string } = {}
): Promise<ConversationListResponse> {
	const baseUrl = getApiBaseUrl();

	const params = new URLSearchParams();
	if (options.limit) params.set('limit', String(options.limit));
	if (options.order) params.set('order', options.order);
	if (options.after) params.set('after', options.after);

	const queryString = params.toString();
	const url = `${baseUrl}/conversations${queryString ? `?${queryString}` : ''}`;

	logger.api.request('GET', url, options);

	const response = await fetch(url);

	logger.api.response('GET', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to list conversations: ${response.status} ${errorText}`);
	}

	const list = (await response.json()) as ConversationListResponse;
	logger.info('api', 'Conversations listed', {
		count: list.data.length,
		hasMore: list.has_more
	});

	return list;
}

/**
 * Create a new conversation on the server.
 *
 * @param metadata - Optional metadata to attach to the conversation
 * @returns The created conversation
 */
export async function createConversation(
	metadata?: Record<string, string>
): Promise<ServerConversation> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations`;

	logger.api.request('POST', url, { metadata });

	const response = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ metadata })
	});

	logger.api.response('POST', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to create conversation: ${response.status} ${errorText}`);
	}

	const conversation = (await response.json()) as ServerConversation;
	logger.info('api', 'Conversation created', { conversationId: conversation.id });

	return conversation;
}

/**
 * Get a conversation by ID.
 *
 * @param id - The conversation ID
 * @param includeItems - Whether to include items in the response
 * @returns The conversation
 */
export async function getConversation(
	id: string,
	includeItems = false
): Promise<ServerConversation> {
	const baseUrl = getApiBaseUrl();
	const params = includeItems ? '?include[]=conversation.items' : '';
	const url = `${baseUrl}/conversations/${id}${params}`;

	logger.api.request('GET', url, { id, includeItems });

	const response = await fetch(url);

	logger.api.response('GET', url, response.status, {});

	if (!response.ok) {
		if (response.status === 404) {
			throw new Error(`Conversation not found: ${id}`);
		}
		const errorText = await response.text();
		throw new Error(`Failed to get conversation: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ServerConversation;
}

/**
 * Update a conversation's metadata.
 *
 * @param id - The conversation ID
 * @param metadata - New metadata
 * @returns The updated conversation
 */
export async function updateConversation(
	id: string,
	metadata: Record<string, string>
): Promise<ServerConversation> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations/${id}`;

	logger.api.request('POST', url, { id, metadata });

	const response = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ metadata })
	});

	logger.api.response('POST', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to update conversation: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ServerConversation;
}

/**
 * Delete a conversation.
 *
 * @param id - The conversation ID
 */
export async function deleteConversation(id: string): Promise<ConversationDeleted> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations/${id}`;

	logger.api.request('DELETE', url, { id });

	const response = await fetch(url, { method: 'DELETE' });

	logger.api.response('DELETE', url, response.status, {});

	if (!response.ok) {
		if (response.status === 404) {
			// Already deleted, treat as success
			return { id, object: 'conversation.deleted', deleted: true };
		}
		const errorText = await response.text();
		throw new Error(`Failed to delete conversation: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ConversationDeleted;
}

/**
 * List items in a conversation.
 *
 * @param conversationId - The conversation ID
 * @param options - Pagination options
 * @returns Paginated list of items
 */
export async function listItems(
	conversationId: string,
	options: PaginationOptions = {}
): Promise<ItemList> {
	const baseUrl = getApiBaseUrl();

	const params = new URLSearchParams();
	if (options.after) params.set('after', options.after);
	if (options.limit) params.set('limit', String(options.limit));
	if (options.order) params.set('order', options.order);
	if (options.include?.length) {
		for (const inc of options.include) {
			params.append('include', inc);
		}
	}

	const queryString = params.toString();
	const url = `${baseUrl}/conversations/${conversationId}/items${queryString ? `?${queryString}` : ''}`;

	logger.api.request('GET', url, { conversationId, ...options });

	const response = await fetch(url);

	logger.api.response('GET', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to list items: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ItemList;
}

/**
 * Add items to a conversation.
 *
 * @param conversationId - The conversation ID
 * @param items - Items to add (max 20)
 * @returns List of added items
 */
export async function createItems(conversationId: string, items: InputItem[]): Promise<ItemList> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations/${conversationId}/items`;

	logger.api.request('POST', url, { conversationId, itemCount: items.length });

	const response = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ items })
	});

	logger.api.response('POST', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to create items: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ItemList;
}

/**
 * Get a single item from a conversation.
 *
 * @param conversationId - The conversation ID
 * @param itemId - The item ID
 * @returns The item
 */
export async function getItem(conversationId: string, itemId: string): Promise<ConversationItem> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations/${conversationId}/items/${itemId}`;

	logger.api.request('GET', url, { conversationId, itemId });

	const response = await fetch(url);

	logger.api.response('GET', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to get item: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ConversationItem;
}

/**
 * Delete an item from a conversation.
 *
 * @param conversationId - The conversation ID
 * @param itemId - The item ID
 * @returns The updated conversation
 */
export async function deleteItem(
	conversationId: string,
	itemId: string
): Promise<ServerConversation> {
	const baseUrl = getApiBaseUrl();
	const url = `${baseUrl}/conversations/${conversationId}/items/${itemId}`;

	logger.api.request('DELETE', url, { conversationId, itemId });

	const response = await fetch(url, { method: 'DELETE' });

	logger.api.response('DELETE', url, response.status, {});

	if (!response.ok) {
		const errorText = await response.text();
		throw new Error(`Failed to delete item: ${response.status} ${errorText}`);
	}

	return (await response.json()) as ServerConversation;
}
