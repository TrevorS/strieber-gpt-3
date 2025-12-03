/**
 * Conversation and Message Types
 *
 * Core types for the conversation state store.
 */

// Import OpenAI response types for use in this file
import type {
	ResponseOutputItem as OpenAIResponseOutputItem,
	ResponseReasoningItem as OpenAIResponseReasoningItem,
	ResponseFunctionWebSearch as OpenAIResponseFunctionWebSearch,
	ResponseCodeInterpreterToolCall as OpenAIResponseCodeInterpreterToolCall,
	ResponseFunctionToolCall as OpenAIResponseFunctionToolCall,
	ResponseOutputMessage as OpenAIResponseOutputMessage,
	ResponseFileSearchToolCall as OpenAIResponseFileSearchToolCall,
	ResponseComputerToolCall as OpenAIResponseComputerToolCall,
	ResponseCustomToolCall as OpenAIResponseCustomToolCall
} from 'openai/resources/responses/responses';

// Re-export with cleaner names for external consumers
export type ResponseOutputItem = OpenAIResponseOutputItem;
export type ResponseReasoningItem = OpenAIResponseReasoningItem;
export type ResponseFunctionWebSearch = OpenAIResponseFunctionWebSearch;
export type ResponseCodeInterpreterToolCall = OpenAIResponseCodeInterpreterToolCall;
export type ResponseFunctionToolCall = OpenAIResponseFunctionToolCall;
export type ResponseOutputMessage = OpenAIResponseOutputMessage;
export type ResponseFileSearchToolCall = OpenAIResponseFileSearchToolCall;
export type ResponseComputerToolCall = OpenAIResponseComputerToolCall;
export type ResponseCustomToolCall = OpenAIResponseCustomToolCall;

/**
 * Generate a UUID v4 using crypto.getRandomValues()
 * Works in all browser contexts (not just secure contexts like crypto.randomUUID)
 */
export function generateUUID(): string {
	const bytes = new Uint8Array(16);
	crypto.getRandomValues(bytes);
	bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
	bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
	const hex = [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

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
	rawOutput?: OpenAIResponseOutputItem[];

	/** Timestamp when message was created */
	createdAt: number;

	/** Whether this message is currently streaming */
	isStreaming?: boolean;
}

// =============================================================================
// Type Guards for Output Items
// =============================================================================

/**
 * Check if an output item is a reasoning item (model thinking/chain-of-thought)
 */
export function isReasoningItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseReasoningItem {
	return item.type === 'reasoning';
}

/**
 * Check if an output item is a message (text output from the model)
 */
export function isMessageItem(item: OpenAIResponseOutputItem): item is OpenAIResponseOutputMessage {
	return item.type === 'message';
}

/**
 * Check if an output item is a function call (custom tool call)
 */
export function isFunctionCallItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseFunctionToolCall {
	return item.type === 'function_call';
}

/**
 * Check if an output item is a web search call
 */
export function isWebSearchItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseFunctionWebSearch {
	return item.type === 'web_search_call';
}

/**
 * Check if an output item is a code interpreter call
 */
export function isCodeInterpreterItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseCodeInterpreterToolCall {
	return item.type === 'code_interpreter_call';
}

/**
 * Check if an output item is a file search call
 */
export function isFileSearchItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseFileSearchToolCall {
	return item.type === 'file_search_call';
}

/**
 * Check if an output item is a computer use call
 */
export function isComputerUseItem(
	item: OpenAIResponseOutputItem
): item is OpenAIResponseComputerToolCall {
	return item.type === 'computer_call';
}

/**
 * Helper to create a new conversation with defaults.
 */
export function createConversation(overrides?: Partial<Conversation>): Conversation {
	const now = Date.now();
	return {
		id: generateUUID(),
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
		id: generateUUID(),
		role,
		content,
		createdAt: Date.now(),
		...overrides
	};
}
