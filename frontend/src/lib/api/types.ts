/**
 * OpenAI API Type Re-exports
 *
 * Re-exports commonly used types from the openai package for convenience.
 * Import from here to avoid deep imports throughout the codebase.
 */

// Core OpenAI client type
export type { OpenAI } from 'openai';
// Error types
export { APIError, AuthenticationError, BadRequestError, RateLimitError } from 'openai';
// Chat Completions types (used by Responses API internally)
export type {
	ChatCompletion,
	ChatCompletionAssistantMessageParam,
	ChatCompletionChunk,
	ChatCompletionContentPart,
	ChatCompletionContentPartImage,
	ChatCompletionContentPartText,
	ChatCompletionCreateParams,
	ChatCompletionCreateParamsNonStreaming,
	ChatCompletionCreateParamsStreaming,
	ChatCompletionMessage,
	ChatCompletionMessageParam,
	ChatCompletionMessageToolCall,
	ChatCompletionRole,
	ChatCompletionSystemMessageParam,
	ChatCompletionTool,
	ChatCompletionToolChoiceOption,
	ChatCompletionToolMessageParam,
	ChatCompletionUserMessageParam
} from 'openai/resources/chat/completions';

// Model types
export type { Model } from 'openai/resources/models';
// Responses API types (OpenAI's newer API)
export type {
	Response,
	ResponseCreateParams,
	ResponseCreateParamsNonStreaming,
	ResponseCreateParamsStreaming
} from 'openai/resources/responses/responses';
