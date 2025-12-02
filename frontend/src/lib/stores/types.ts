/**
 * Conversation and Message Types
 *
 * Core types for the conversation state store.
 */

/**
 * A chat conversation containing messages and metadata.
 */
export interface Conversation {
	/** Unique identifier (UUID) */
	id: string;

	/** Display title (auto-generated or user-edited) */
	title: string;

	/** Timestamp when conversation was created */
	createdAt: number;

	/** Timestamp when conversation was last updated */
	updatedAt: number;

	/** Last response ID for context chaining with Responses API */
	lastResponseId: string | null;

	/** Messages in chronological order */
	messages: Message[];
}

/**
 * A single message in a conversation.
 */
export interface Message {
	/** Unique identifier (UUID) */
	id: string;

	/** Who sent the message */
	role: 'user' | 'assistant';

	/** Text content (may include markdown) */
	content: string;

	/** Raw output items from Responses API (tool calls, reasoning, etc.) */
	rawOutput?: OutputItem[];

	/** Timestamp when message was created */
	createdAt: number;

	/** Whether this message is currently streaming */
	isStreaming?: boolean;
}

/**
 * Output item from Responses API.
 * Simplified version - expand as needed for tool displays.
 */
export interface OutputItem {
	/** Item type (message, function_call, reasoning, etc.) */
	type: string;

	/** Item ID */
	id?: string;

	/** Content varies by type */
	content?: unknown;
}

/**
 * Helper to create a new conversation with defaults.
 */
export function createConversation(overrides?: Partial<Conversation>): Conversation {
	const now = Date.now();
	return {
		id: crypto.randomUUID(),
		title: 'New Chat',
		createdAt: now,
		updatedAt: now,
		lastResponseId: null,
		messages: [],
		...overrides
	};
}

/**
 * Helper to create a new message with defaults.
 */
export function createMessage(
	role: Message['role'],
	content: string,
	overrides?: Partial<Message>
): Message {
	return {
		id: crypto.randomUUID(),
		role,
		content,
		createdAt: Date.now(),
		...overrides
	};
}
